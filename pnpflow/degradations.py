import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter

from pnpflow.utils import square_mask, random_mask, paintbrush_mask, gaussian_blur, gaussian_2d_kernel, downsample, upsample, bicubic_filter, create_downsampling_matrix


class Degradation:

    def H(self, x):
        raise NotImplementedError()

    def H_adj(self, x):
        raise NotImplementedError()

class BaseDegradation(torch.nn.Module):
    def __init__(self, noise_std=0.0):
        super().__init__()
        self.noise_std = noise_std

    def forward(self, x):
        x = x + self.noise_std * torch.randn_like(x)
        return x

    def pseudo_inv(self, y):
        return y


def zero_filler(x, scale):
    B, C, H, W = x.shape
    scale = int(scale)
    H_new, W_new = H * scale, W * scale
    out = torch.zeros(B, C, H_new, W_new, dtype=x.dtype, device=x.device)
    out[:, :, ::scale, ::scale] = x
    return out


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def norm(lst: list) -> float:
    if not isinstance(lst, list):
        raise ValueError("Norm takes a list as its argument")

    if lst == []:
        return 0

    return (sum((i**2 for i in lst)))**0.5


def polar2z(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return r * np.exp(1j * theta)


class Kernel(object):
    def __init__(self, size: tuple = (100, 100), intensity: float = 0):
        if not isinstance(size, tuple):
            raise ValueError("Size must be TUPLE of 2 positive integers")
        elif len(size) != 2 or type(size[0]) != type(size[1]) != int:
            raise ValueError("Size must be tuple of 2 positive INTEGERS")
        elif size[0] < 0 or size[1] < 0:
            raise ValueError("Size must be tuple of 2 POSITIVE integers")

        if type(intensity) not in [int, float, np.float32, np.float64]:
            raise ValueError("Intensity must be a number between 0 and 1")
        elif intensity < 0 or intensity > 1:
            raise ValueError("Intensity must be a number between 0 and 1")

        self.SIZE = size
        self.INTENSITY = float(intensity)
        self.SIZEx2 = tuple([2 * i for i in size])
        self.x, self.y = self.SIZEx2
        self.DIAGONAL = (self.x**2 + self.y**2)**0.5
        self.kernel_is_generated = False

    def _createPath(self):
        def getSteps():
            self.MAX_PATH_LEN = 0.75 * self.DIAGONAL * (np.random.uniform() + np.random.uniform(0, self.INTENSITY**2))
            steps = []
            while sum(steps) < self.MAX_PATH_LEN:
                step = np.random.beta(1, 30) * (1 - self.INTENSITY + 0.1) * self.DIAGONAL
                if step < self.MAX_PATH_LEN:
                    steps.append(step)
            if len(steps) == 0:
                steps.append(self.DIAGONAL * 0.05)
            self.NUM_STEPS = len(steps)
            self.STEPS = np.asarray(steps)

        def getAngles():
            self.MAX_ANGLE = np.random.uniform(0, self.INTENSITY * np.pi)
            self.JITTER = np.random.beta(2, 20)
            angles = [np.random.uniform(low=-self.MAX_ANGLE, high=self.MAX_ANGLE)]
            while len(angles) < self.NUM_STEPS:
                angle = np.random.triangular(0, self.INTENSITY * self.MAX_ANGLE, self.MAX_ANGLE + 0.1)
                if np.random.uniform() < self.JITTER:
                    angle *= -np.sign(angles[-1])
                else:
                    angle *= np.sign(angles[-1])
                angles.append(angle)
            self.ANGLES = np.asarray(angles)

        getSteps()
        getAngles()
        complex_increments = polar2z(self.STEPS, self.ANGLES)
        self.path_complex = np.cumsum(complex_increments)
        self.com_complex = sum(self.path_complex) / self.NUM_STEPS
        center_of_kernel = (self.x + 1j * self.y) / 2
        self.path_complex -= self.com_complex
        self.path_complex *= np.exp(1j * np.random.uniform(0, np.pi))
        self.path_complex += center_of_kernel
        self.path = [(i.real, i.imag) for i in self.path_complex]

    def _createKernel(self):
        if self.kernel_is_generated:
            return None

        self._createPath()
        kernel_image = Image.new("RGB", self.SIZEx2)
        painter = ImageDraw.Draw(kernel_image)
        painter.line(xy=self.path, width=int(self.DIAGONAL / 150))
        kernel_image = kernel_image.filter(ImageFilter.GaussianBlur(radius=int(self.DIAGONAL * 0.01)))
        kernel_image = kernel_image.resize(self.SIZE, resample=Image.LANCZOS)
        kernel_image = kernel_image.convert("L")
        self.kernel_image = kernel_image
        self.kernel_is_generated = True

    @property
    def kernelMatrix(self) -> np.ndarray:
        self._createKernel()
        kernel = np.asarray(self.kernel_image, dtype=np.float32)
        kernel /= np.sum(kernel)
        return kernel


class Denoising(Degradation):
    def H(self, x):
        return x

    def H_adj(self, x):
        return x


class BoxInpainting(Degradation):
    def __init__(self, half_size_mask):
        super().__init__()
        self.half_size_mask = half_size_mask

    def H(self, x):
        return square_mask(x, self.half_size_mask)

    def H_adj(self, x):
        return square_mask(x, self.half_size_mask)


class RandomInpainting(Degradation):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def H(self, x):
        return random_mask(x, self.p)

    def H_adj(self, x):
        return random_mask(x, self.p)


class PaintbrushInpainting(Degradation):
    def H(self, x):
        return paintbrush_mask(x)

    def H_adj(self, x):
        return paintbrush_mask(x)


class GaussianDeblurring(Degradation):
    def __init__(self, sigma_blur, kernel_size,  mode="fft", num_channels=3, dim_image=128, device="cuda") -> None:
        super().__init__()
        self.mode = mode
        self.sigma = sigma_blur
        self.kernel_size = kernel_size
        self.kernel = gaussian_2d_kernel(sigma_blur, kernel_size).to(device)
        filter = torch.zeros(
            (1, num_channels) + (dim_image, dim_image), device=device
        )

        filter[..., : kernel_size, : kernel_size] = self.kernel
        self.filter = torch.roll(
            filter, shifts=(-(kernel_size-1)//2, -(kernel_size-1)//2), dims=(2, 3))
        self.device = device

    def H(self, x):
        if self.mode != "fft":
            kernel = self.kernel.view(
                1, 1, self.kernel_size,  self.kernel_size)
            kernel = self.kernel.repeat(x.shape[1], 1, 1, 1)
            return F.conv2d(x, kernel, stride=1, padding='same', groups=x.shape[1])
        else:
            return torch.real(torch.fft.ifft2(
                torch.fft.fft2(x.to(self.device)) * torch.fft.fft2(self.filter)))

    def H_adj(self, x):
        if self.mode != "fft":
            kernel = self.kernel.view(
                1, 1, self.kernel_size,  self.kernel_size)
            kernel = self.kernel.repeat(x.shape[1], 1, 1, 1)
            return F.conv2d(x, kernel, stride=1, padding='same', groups=x.shape[1])
        else:
            return torch.real(torch.fft.ifft2(
                torch.fft.fft2(x.to(self.device)) * torch.conj(torch.fft.fft2(self.filter))))


def _motion_kernel(kernel_size: int, intensity: float, device=None) -> torch.Tensor:
    kernel = Kernel(size=(kernel_size, kernel_size), intensity=float(np.clip(intensity, 0.0, 1.0))).kernelMatrix
    return torch.from_numpy(kernel).to(device=device, dtype=torch.float32)


class MotionBlur(Degradation):
    def __init__(
        self,
        kernel_size=21,
        angle=0.0,
        mode="fft",
        num_channels=3,
        dim_image=256,
        device="cuda",
        noise_std=0.01,
        img_size=None,
    ) -> None:
        super().__init__()
        self.mode = mode
        # Keep the existing argument name for compatibility with current call sites.
        self.intensity = float(np.clip(angle, 0.0, 1.0))
        self.kernel_size = kernel_size
        self.device = device
        self.noise_std = noise_std
        if img_size is not None:
            dim_image = img_size
        self.kernel = _motion_kernel(kernel_size, self.intensity, device=device)
        self.num_channels = num_channels
        self.dim_image = dim_image
        self.filter = self._build_filter(dim_image, device)

    def _build_filter(self, dim_image, device):
        filter = torch.zeros((1, self.num_channels) + (dim_image, dim_image), device=device)
        filter[..., : self.kernel_size, : self.kernel_size] = self.kernel.view(1, 1, self.kernel_size, self.kernel_size)
        return torch.roll(
            filter, shifts=(-(self.kernel_size - 1) // 2, -(self.kernel_size - 1) // 2), dims=(2, 3)
        )

    def _ensure_filter(self, x):
        dim_image = x.shape[-1]
        if self.filter.shape[-1] != dim_image or self.filter.device != x.device:
            self.filter = self._build_filter(dim_image, x.device)
        return self.filter

    def H(self, x):
        if self.mode != "fft":
            kernel = self.kernel.view(1, 1, self.kernel_size, self.kernel_size)
            kernel = kernel.repeat(x.shape[1], 1, 1, 1)
            return F.conv2d(x, kernel, stride=1, padding="same", groups=x.shape[1])
        filter = self._ensure_filter(x)
        return torch.real(torch.fft.ifft2(torch.fft.fft2(x.to(filter.device)) * torch.fft.fft2(filter)))

    def H_adj(self, x):
        if self.mode != "fft":
            kernel = torch.flip(self.kernel, dims=(0, 1)).view(1, 1, self.kernel_size, self.kernel_size)
            kernel = kernel.repeat(x.shape[1], 1, 1, 1)
            return F.conv2d(x, kernel, stride=1, padding="same", groups=x.shape[1])
        filter = self._ensure_filter(x)
        return torch.real(torch.fft.ifft2(torch.fft.fft2(x.to(filter.device)) * torch.conj(torch.fft.fft2(filter))))


class Superresolution(Degradation):
    def __init__(self, sf, dim_image, mode=None, device="cuda") -> None:
        super().__init__()
        self.sf = sf
        self.mode = mode
        if mode == "bicubic":
            self.filter = torch.nn.Parameter(
                bicubic_filter(sf), requires_grad=False
            ).to(device)
            # Move batch dim of the input into channels

            filter = torch.zeros(
                (1, 3) + (dim_image, dim_image), device=device)

            filter[..., : self.filter.shape[-1],
                   : self.filter.shape[-1]] = self.filter
            self.filter = torch.roll(
                filter, shifts=(-(self.filter.shape[-1]-1)//2, -(self.filter.shape[-1]-1)//2), dims=(2, 3))
        self.downsampling_matrix = create_downsampling_matrix(
            dim_image, dim_image, sf, device)

    def H(self, x):

        if self.mode == None:
            return downsample(x, self.sf)
        elif self.mode == "bicubic":
            x_ = torch.real(torch.fft.ifft2(
                torch.fft.fft2(x) * torch.fft.fft2(self.filter)))
            return downsample(x_, self.sf)

    def H_adj(self, x):
        if self.mode == None:
            return upsample(x, self.sf)
        elif self.mode == "bicubic":
            x_ = upsample(x, self.sf)
            return torch.real(torch.fft.ifft2(torch.fft.fft2(x_) * torch.conj(torch.fft.fft2(self.filter))))
