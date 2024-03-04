import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
import os
import cv2
from PIL import Image


def gaussian_weight(ksize, sigma=None):
    if sigma is None:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8

    center = ksize // 2
    x = (np.arange(ksize, dtype=np.float32) - center)
    kernel_1d = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = kernel_1d[..., None] @ kernel_1d[None, ...]
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum()  # Normalization
    return kernel


def gaussian_filter(img, ksize, sigma=None):
    kernel = gaussian_weight(ksize, sigma)
    kernel = kernel.view(1, 1, ksize, ksize).repeat(img.shape[1], 1, 1, 1)
    pad = (ksize - 1) // 2
    img = F.conv2d(img, weight=kernel, stride=1, padding=pad, groups=img.shape[1])
    return img


def bilateral_filter(img, ksize, sigma_space=None, sigma_density=None):
    device = img.device
    if sigma_space is None:
        sigma_space = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    if sigma_density is None:
        sigma_density = sigma_space

    pad = (ksize - 1) // 2
    pad_img = F.pad(img, pad=[pad, pad, pad, pad], mode='reflect')
    pad_img_patches = pad_img.unfold(2, ksize, 1).unfold(3, ksize, 1)
    patch_dim = pad_img_patches.dim()

    diff_density = pad_img_patches - img.unsqueeze(-1).unsqueeze(-1)
    weight_density = torch.exp(-(diff_density ** 2) / (2 * sigma_density ** 2))
    weight_density /= weight_density.sum(dim=(-1, -2), keepdim=True)

    weight_space = gaussian_weight(ksize, sigma_space).to(device=device)
    weight_space_dim = (patch_dim - 2) * (1,) + (ksize, ksize)
    weight_space = weight_space.view(*weight_space_dim).expand_as(weight_density)

    weight = weight_density * weight_space
    weight_sum = weight.sum(dim=(-1, -2))
    img = (weight * pad_img_patches).sum(dim=(-1, -2)) / weight_sum
    return img


######################################################
######################################################
######################################################

class GaussianFilter(nn.Module):
    def __init__(self, ksize=5, sigma=None):
        super(GaussianFilter, self).__init__()
        # initialize guassian kernel
        if sigma is None:
            sigma = 0.3 * ((ksize - 1) / 2.0 - 1) + 0.8
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(ksize)
        x_grid = x_coord.repeat(ksize).view(ksize, ksize)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        # Calculate the 2-dimensional gaussian kernel
        center = ksize // 2
        weight = torch.exp(-torch.sum((xy_grid - center) ** 2., dim=-1) / (2 * sigma ** 2))
        # Make sure sum of values in gaussian kernel equals 1.
        weight /= torch.sum(weight)
        self.gaussian_weight = weight

        # Reshape to 2d depthwise convolutional weight
        weight = weight.view(1, 1, ksize, ksize)
        weight = weight.repeat(3, 1, 1, 1)

        # create gaussian filter as convolutional layer
        pad = (ksize - 1) // 2
        self.filter = nn.Conv2d(3, 3, ksize, stride=1, padding=pad, groups=3, bias=False, padding_mode='reflect')
        self.filter.weight.data = weight
        self.filter.weight.requires_grad = False

    def forward(self, x):
        return self.filter(x)


class BilateralFilter(nn.Module):
    def __init__(self, ksize=5, sigma_space=None, sigma_density=None):
        super(BilateralFilter, self).__init__()
        # initialization
        if sigma_space is None:
            self.sigma_space = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        if sigma_density is None:
            self.sigma_density = self.sigma_space

        self.pad = (ksize - 1) // 2
        self.ksize = ksize
        # get the spatial gaussian weight
        self.weight_space = GaussianFilter(ksize=self.ksize, sigma=self.sigma_space).gaussian_weight.cuda()

    def forward(self, x):
        # Extracts sliding local patches from a batched input tensor.
        x_pad = F.pad(x, pad=[self.pad, self.pad, self.pad, self.pad], mode='reflect')
        x_patches = x_pad.unfold(2, self.ksize, 1).unfold(3, self.ksize, 1)
        patch_dim = x_patches.dim()

        # Calculate the 2-dimensional gaussian kernel
        diff_density = x_patches - x.unsqueeze(-1).unsqueeze(-1)
        weight_density = torch.exp(-(diff_density ** 2) / (2 * self.sigma_density ** 2))
        # Normalization
        weight_density /= weight_density.sum(dim=(-1, -2), keepdim=True)
        # print(weight_density.shape)

        # Keep same shape with weight_density
        weight_space_dim = (patch_dim - 2) * (1,) + (self.ksize, self.ksize)
        weight_space = self.weight_space.view(*weight_space_dim).expand_as(weight_density)

        # get the final kernel weight
        weight = weight_density * weight_space
        weight_sum = weight.sum(dim=(-1, -2))
        x = (weight * x_patches).sum(dim=(-1, -2)) / weight_sum
        return x


if __name__ == "__main__":
    # a = torch.randn(1, 3, 6, 6)
    # b = torch.randn(1, 3, 64, 64)
    # b_filter = BilateralFilter()
    # # output_a = gaussian_filter(a, ksize=5)
    # # output_a2 = g_filter(a)
    # # output_b = bilateral_filter(b, ksize=5)
    # output_b = b_filter(b)

    # print(output_b.shape)

    img = Image.open('./0001x4w3.png').convert('RGB')
    img = ToTensor()(img).unsqueeze(0)

    b_filter = BilateralFilter()
    output = b_filter(img)

    output = output.detach().squeeze(0).cpu()
    ToPILImage()(output).save('./test.png')