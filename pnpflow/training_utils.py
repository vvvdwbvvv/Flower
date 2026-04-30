import os
from pathlib import Path
from typing import Optional

import torch


def optional_value(value):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in ("none", "null", ""):
        return None
    return value


def get_amp_settings(args, device: str):
    dtype_name = str(getattr(args, "amp_dtype", "none")).lower()
    enabled = bool(getattr(args, "amp", False)) and dtype_name != "none"
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    if not enabled:
        return False, None, None
    if dtype_name == "bf16":
        dtype = torch.bfloat16
    elif dtype_name in ("fp16", "float16"):
        dtype = torch.float16
    else:
        raise ValueError("amp_dtype must be one of: none, fp16, bf16")

    scaler = torch.cuda.amp.GradScaler(enabled=device_type == "cuda" and dtype == torch.float16)
    return True, dtype, scaler


def autocast_context(enabled: bool, dtype, device: str):
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    return torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=enabled)


def checkpoint_dir(model_path: str) -> str:
    path = os.path.join(model_path, "checkpoints")
    os.makedirs(path, exist_ok=True)
    return path


def checkpoint_path(model_path: str, epoch: int) -> str:
    return os.path.join(checkpoint_dir(model_path), f"checkpoint_epoch_{epoch:04d}.pt")


def latest_checkpoint(model_path: str) -> Optional[str]:
    ckpt_dir = os.path.join(model_path, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    checkpoints = sorted(Path(ckpt_dir).glob("checkpoint_epoch_*.pt"))
    if not checkpoints:
        return None
    return str(checkpoints[-1])


def save_training_checkpoint(
    path: str,
    model,
    optimizer,
    epoch: int,
    global_step: int,
    args,
    scaler=None,
    scheduler=None,
):
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": dict(args),
    }
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)
    torch.save(payload, os.path.join(os.path.dirname(path), "latest.pt"))


def load_training_checkpoint(path: str, model, optimizer=None, scaler=None, scheduler=None, device="cuda"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def resolve_resume_checkpoint(args, model_path: str) -> Optional[str]:
    resume_checkpoint = optional_value(getattr(args, "resume_checkpoint", None))
    if resume_checkpoint:
        return resume_checkpoint
    if bool(getattr(args, "resume", False)):
        return latest_checkpoint(model_path)
    return None
