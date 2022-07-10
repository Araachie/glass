import torch
import torch.nn as nn


class FinalBlock(nn.Module):
    """
    Final block transforming features into an image
    """

    def __init__(self, in_planes: int, out_planes: int, kernel_size: int = 3, padding: int = 1):
        """

        :param in_planes: Input features to the module
        :param out_planes: Output feature
        """

        super(FinalBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=(kernel_size, kernel_size),
            stride=(1, 1),
            padding=(padding, padding),
            bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.tanh(x)

        return x
