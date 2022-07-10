# The modulated conv2d code adopted from this repository https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/networks.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulated_conv2d(
    x,                  # Input tensor: [batch_size, in_channels, in_height, in_width]
    w,                  # Weight tensor: [out_channels, in_channels, kernel_height, kernel_width]
    s,                  # Style tensor: [batch_size, in_channels]
    demodulate  = True, # Apply weight demodulation?
    padding     = 0,    # Padding: int or [padH, padW]
    input_gain  = None, # Optional scale factors for the input channels: [], [in_channels], or [batch_size, in_channels]
):
    batch_size = int(x.shape[0])
    out_channels, in_channels, kh, kw = w.shape

    # Pre-normalize inputs.
    if demodulate:
        w = w * w.square().mean([1,2,3], keepdim=True).rsqrt()
        s = s * s.square().mean().rsqrt()

    # Modulate weights.
    w = w.unsqueeze(0) # [NOIkk]
    w = w * s.unsqueeze(1).unsqueeze(3).unsqueeze(4) # [NOIkk]

    # Demodulate weights.
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
        w = w * dcoefs.unsqueeze(2).unsqueeze(3).unsqueeze(4) # [NOIkk]

    # Apply input scaling.
    if input_gain is not None:
        input_gain = input_gain.expand(batch_size, in_channels) # [NI]
        w = w * input_gain.unsqueeze(1).unsqueeze(3).unsqueeze(4) # [NOIkk]

    # Execute as one fused op using grouped convolution.
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = torch.nn.functional.conv2d(input=x, weight=w.to(x.dtype), padding=padding, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    return x


class ModulatedConvBlock(nn.Module):
    """
    Modulated convolutional block with normalization and activation
    """
    expansion = 1

    def __init__(self, in_planes, out_planes, in_style_size, downsample_factor=1, drop_final_activation=False):
        """

        :param in_planes: Input features to the module
        :param out_planes: Output feature
        :param downsample_factor: Reduction factor in feature dimension
        :param drop_final_activation: if True does not pass the final output through the activation function
        """

        super(ModulatedConvBlock, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.in_style_size = in_style_size

        norm_layer = nn.BatchNorm2d

        self.downsample_factor = downsample_factor
        self.drop_final_activation = drop_final_activation

        self.affine = nn.Linear(in_style_size, in_planes)
        self.weight = nn.Parameter(torch.randn([out_planes, in_planes, 3, 3]))

        self.bn1 = norm_layer(out_planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """

        :param x: (b, c, h, w)
        :param w: (b, c)
        :return:
        """

        out = modulated_conv2d(x=x, w=self.weight, s=self.affine(w), padding=1)
        # Downscale if required
        if self.downsample_factor != 1:
            out = F.avg_pool2d(out, self.downsample_factor)
        out = self.bn1(out)
        # Applies activation if required
        if not self.drop_final_activation:
            out = self.relu(out)

        return out
