import torch
import torch.nn.functional as F
from pnpflow.utils import square_mask, random_mask, paintbrush_mask, gaussian_blur, gaussian_2d_kernel, downsample, upsample, bicubic_filter, create_downsampling_matrix


def motion_blur_kernel(kernel_size, angle=0.0, device="cpu"):
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd for motion blur.")

    coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    center = (kernel_size - 1) / 2.0

    xx = xx - center
    yy = yy - center

    theta = torch.tensor(angle * torch.pi / 180.0,
                         dtype=torch.float32, device=device)
    direction_x = torch.cos(theta)
    direction_y = torch.sin(theta)

    along = xx * direction_x + yy * direction_y
    perp = -xx * direction_y + yy * direction_x

    kernel = ((along.abs() <= center) & (perp.abs() <= 0.5)).float()
    kernel_sum = kernel.sum()
    if kernel_sum == 0:
        raise ValueError("Motion blur kernel is empty. Check kernel_size and angle.")
    return kernel / kernel_sum


class Degradation:

    def H(self, x):
        raise NotImplementedError()

    def H_adj(self, x):
        raise NotImplementedError()


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


class MotionDeblurring(Degradation):
    def __init__(self, kernel_size, angle=0.0, mode="fft", num_channels=3, dim_image=128, device="cuda") -> None:
        super().__init__()
        self.mode = mode
        self.kernel_size = kernel_size
        self.angle = angle
        self.device = device
        self.kernel = motion_blur_kernel(
            kernel_size, angle=angle, device=device)

        filter = torch.zeros(
            (1, num_channels) + (dim_image, dim_image), device=device
        )
        filter[..., : kernel_size, : kernel_size] = self.kernel
        self.filter = torch.roll(
            filter, shifts=(-(kernel_size - 1) // 2, -(kernel_size - 1) // 2), dims=(2, 3))

        flipped_kernel = torch.flip(self.kernel, dims=(0, 1))
        adjoint_filter = torch.zeros(
            (1, num_channels) + (dim_image, dim_image), device=device
        )
        adjoint_filter[..., : kernel_size, : kernel_size] = flipped_kernel
        self.adjoint_filter = torch.roll(
            adjoint_filter, shifts=(-(kernel_size - 1) // 2, -(kernel_size - 1) // 2), dims=(2, 3))

    def H(self, x):
        if self.mode != "fft":
            kernel = self.kernel.view(1, 1, self.kernel_size, self.kernel_size)
            kernel = kernel.repeat(x.shape[1], 1, 1, 1)
            return F.conv2d(x, kernel, stride=1, padding='same', groups=x.shape[1])
        return torch.real(torch.fft.ifft2(
            torch.fft.fft2(x.to(self.device)) * torch.fft.fft2(self.filter)))

    def H_adj(self, x):
        if self.mode != "fft":
            kernel = torch.flip(self.kernel, dims=(0, 1)).view(
                1, 1, self.kernel_size, self.kernel_size)
            kernel = kernel.repeat(x.shape[1], 1, 1, 1)
            return F.conv2d(x, kernel, stride=1, padding='same', groups=x.shape[1])
        return torch.real(torch.fft.ifft2(
            torch.fft.fft2(x.to(self.device)) * torch.fft.fft2(self.adjoint_filter)))


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
