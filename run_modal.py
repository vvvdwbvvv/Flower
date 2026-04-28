"""
Flower Modal Cloud Training Runner.

Runs the existing `main.py --opts ...` training entrypoint on a Modal GPU while
keeping datasets, checkpoints, and results in Modal volumes.

Setup (one-time):
    modal setup
    modal volume create flower-data
    modal volume create flower-model
    modal volume create flower-results
    modal volume put flower-data ./data/ /

After upload, the data volume root should contain dataset directories such as
`afhq_cat/` or `gopro/`. For GoPro, the volume root may also be the dataset
root directly, containing `train/` and `test/`.

Example:
    modal run run_modal.py --dataset afhq_cat --num-epoch 400 --batch-size-train 12
    modal run run_modal.py --dataset gopro --num-epoch 200 --batch-size-train 24

Notes:
    - Source code is copied from this repository into the image.
    - Dataset files are expected at /workspace/Flower/data inside the container.
    - For GoPro, missing data is downloaded into the data volume by default.
    - Checkpoints are saved to the flower-model volume.
    - Training logs, samples, and FID text files are saved to flower-results.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import time
import os
from pathlib import Path

import modal

app = modal.App("flower-train")

PYTORCH_IMAGE = "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"
PROJECT_DIR = "/workspace/Flower"

DATA_VOLUME_NAME = "flower-data"
MODEL_VOLUME_NAME = "flower-model"
RESULTS_VOLUME_NAME = "flower-results"

DATA_PATH = f"{PROJECT_DIR}/data"
MODEL_PATH = f"{PROJECT_DIR}/model"
RESULTS_PATH = f"{PROJECT_DIR}/results"

DEFAULT_GPU = "L40S:1"
DEFAULT_TIMEOUT_SECONDS = 24 * 60 * 60
DEFAULT_DATASET = "gopro"
DEFAULT_MODEL = "ot"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_BATCH_SIZE_TRAIN = 24
DEFAULT_NUM_EPOCH = 200
DEFAULT_LR = 1e-4

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(PYTORCH_IMAGE, add_python="3.11")
    .apt_install("git", "wget", "unzip", "libgl1", "libglib2.0-0")
    .pip_install(
        "autograd==1.6.2",
        "Cython==3.0.10",
        "deepinv==0.2.0",
        "diffeqtorch==1.0.0",
        "gdown",
        "h5py",
        "kaggle",
        "lightning",
        "lpips",
        "matplotlib",
        "ml_collections",
        "ninja",
        "numdifftools",
        "opencv-python",
        "pandas",
        "Pillow",
        "POT",
        "pytorch-ignite",
        "pytorch-lightning",
        "scikit-image",
        "scikit-learn",
        "scipy",
        "sympy",
        "tensorboard",
        "toolz",
        "torchdiffeq",
        "torchmetrics",
        "tqdm",
        "urllib3",
        "wget",
    )
    .env(
        {
            "MPLBACKEND": "Agg",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": PROJECT_DIR,
            "GOPRO_ROOT": f"{DATA_PATH}/gopro",
        }
    )
    .add_local_dir(
        ".",
        remote_path=PROJECT_DIR,
        ignore=[
            ".git",
            "__pycache__",
            ".pytest_cache",
            "data",
            "model",
            "results",
            "results_laplace",
        ],
    )
)


def build_train_opts(
    dataset: str = DEFAULT_DATASET,
    model: str = DEFAULT_MODEL,
    batch_size_train: int = DEFAULT_BATCH_SIZE_TRAIN,
    num_epoch: int = DEFAULT_NUM_EPOCH,
    lr: float = DEFAULT_LR,
    device: str = DEFAULT_DEVICE,
    extra_opts: list[str] | None = None,
) -> list[str]:
    """Build the `--opts` list consumed by main.py."""
    opts = [
        "dataset",
        dataset,
        "train",
        "True",
        "eval",
        "False",
        "compute_metrics",
        "False",
        "batch_size_train",
        str(batch_size_train),
        "num_epoch",
        str(num_epoch),
        "lr",
        str(lr),
        "model",
        model,
        "device",
        device,
    ]
    if extra_opts:
        opts.extend(extra_opts)
    return opts


def build_equivalent_cli(opts: list[str]) -> str:
    """Return the equivalent local training command."""
    return "python main.py --opts " + " ".join(shlex.quote(opt) for opt in opts)


def get_opt(opts: list[str], key: str, default: str | None = None) -> str | None:
    """Read one value from a flat `--opts key value ...` list."""
    try:
        return opts[opts.index(key) + 1]
    except (ValueError, IndexError):
        return default


def short_dir_listing(path: Path) -> str:
    """Return a compact directory listing for error messages."""
    if not path.exists():
        return "<missing>"
    entries = sorted(child.name for child in path.iterdir())
    if len(entries) > 20:
        entries = entries[:20] + ["..."]
    return ", ".join(entries) if entries else "<empty>"


def find_dataset_root(dataset: str) -> Path | None:
    """Find expected or accidentally nested dataset directories."""
    data_path = Path(DATA_PATH)
    candidates = [data_path / dataset, data_path / "data" / dataset]

    if dataset == "gopro":
        for candidate in [data_path, *candidates]:
            if (candidate / "train").is_dir() and (candidate / "test").is_dir():
                return candidate
        return None

    if dataset == "afhq_cat":
        for candidate in candidates:
            if (candidate / "train" / "cat").is_dir() and (candidate / "test" / "cat").is_dir():
                return candidate
        return None

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def ensure_dataset(dataset: str, auto_download_data: bool) -> dict[str, str]:
    """Validate dataset availability and optionally bootstrap GoPro."""
    data_volume.reload()
    dataset_root = find_dataset_root(dataset)
    if dataset_root is not None:
        if dataset == "gopro":
            return {"GOPRO_ROOT": str(dataset_root)}
        return {}

    print(f"Dataset '{dataset}' was not found under {DATA_PATH}.")
    print(f"Top-level data volume entries: {short_dir_listing(Path(DATA_PATH))}")

    if dataset == "gopro" and auto_download_data:
        print("Downloading GoPro into the flower-data volume with download_data.sh...")
        completed = subprocess.run(
            ["bash", "download_data.sh"],
            cwd=PROJECT_DIR,
            env={**os.environ, "DOWNLOAD_GOPRO": "1"},
            check=False,
        )
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, completed.args)

        data_volume.commit()
        data_volume.reload()
        dataset_root = find_dataset_root(dataset)
        if dataset_root is not None:
            return {"GOPRO_ROOT": str(dataset_root)}

    upload_hint = (
        f"Upload data so the volume root contains '{dataset}/'. For example:\n"
        f"  modal volume put {DATA_VOLUME_NAME} ./data/ /\n"
        f"Expected path after mounting: {DATA_PATH}/{dataset}\n"
        f"For GoPro, {DATA_PATH}/train and {DATA_PATH}/test are also accepted."
    )
    raise FileNotFoundError(upload_hint)


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT_SECONDS,
    volumes={
        DATA_PATH: data_volume,
        MODEL_PATH: model_volume,
        RESULTS_PATH: results_volume,
    },
)
def run_training(opts: list[str], auto_download_data: bool = True) -> dict:
    """Run Flower training on a Modal GPU and persist artifacts to volumes."""
    command = [sys.executable, "main.py", "--opts", *opts]
    started_at = time.time()
    dataset = get_opt(opts, "dataset", DEFAULT_DATASET) or DEFAULT_DATASET

    print("Running training command:")
    print(" ".join(shlex.quote(part) for part in command))
    print(f"Working directory: {PROJECT_DIR}")
    print(f"Data volume: {DATA_VOLUME_NAME} -> {DATA_PATH}")
    print(f"Model volume: {MODEL_VOLUME_NAME} -> {MODEL_PATH}")
    print(f"Results volume: {RESULTS_VOLUME_NAME} -> {RESULTS_PATH}")

    env_updates = ensure_dataset(dataset, auto_download_data)
    if env_updates:
        print("Dataset environment overrides:")
        for key, value in env_updates.items():
            print(f"  {key}={value}")

    completed = subprocess.run(
        command,
        cwd=PROJECT_DIR,
        env={**os.environ, **env_updates},
        check=False,
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)

    model_volume.commit()
    results_volume.commit()

    return {
        "command": build_equivalent_cli(opts),
        "elapsed_seconds": round(time.time() - started_at, 2),
        "model_path": MODEL_PATH,
        "results_path": RESULTS_PATH,
    }


@app.local_entrypoint()
def main(
    dataset: str = DEFAULT_DATASET,
    model: str = DEFAULT_MODEL,
    batch_size_train: int = DEFAULT_BATCH_SIZE_TRAIN,
    num_epoch: int = DEFAULT_NUM_EPOCH,
    lr: float = DEFAULT_LR,
    device: str = DEFAULT_DEVICE,
    extra_opts: str = "",
    auto_download_data: bool = True,
):
    """Launch Flower training on Modal."""
    parsed_extra_opts = shlex.split(extra_opts) if extra_opts else []
    opts = build_train_opts(
        dataset=dataset,
        model=model,
        batch_size_train=batch_size_train,
        num_epoch=num_epoch,
        lr=lr,
        device=device,
        extra_opts=parsed_extra_opts,
    )

    print("Equivalent local command:")
    print(build_equivalent_cli(opts))
    print(f"\nLaunching Modal training on {DEFAULT_GPU}...")

    result = run_training.remote(opts, auto_download_data)
    print("\nTraining finished.")
    print(f"Elapsed: {result['elapsed_seconds']} seconds")
    print(f"Checkpoints: modal volume {MODEL_VOLUME_NAME}:{MODEL_PATH}")
    print(f"Results: modal volume {RESULTS_VOLUME_NAME}:{RESULTS_PATH}")
