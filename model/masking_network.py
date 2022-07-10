import torch
import torch.nn as nn
import torch.nn.functional as F


from model.layers import ResidualBlock
from model.layers import UpBlock
from model.layers.utils import SequenceConverter

from utils.configuration import Configuration
from utils.dict_wrapper import DictWrapper


class MaskingNetwork(nn.Module):
    """
    Model that predicts masks
    """

    def __init__(self, in_channels: int):
        super(MaskingNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)

        residual_blocks = [
            ResidualBlock(64, 256, downsample_factor=2),  # res / 2
            ResidualBlock(256, 256, downsample_factor=2),  # res / 4
            ResidualBlock(256, 256, downsample_factor=1),  # res / 4
            ResidualBlock(256, 256, downsample_factor=1),  # res / 4
            ResidualBlock(256, 256, downsample_factor=1),  # res / 4
            ResidualBlock(256, 256, downsample_factor=1),  # res / 4
            ResidualBlock(256, 256, downsample_factor=1),  # res / 4
            ResidualBlock(256, 256, downsample_factor=1),  # res / 4
            ResidualBlock(256, 256, downsample_factor=1),  # res / 4
            ResidualBlock(256, 256, downsample_factor=1),  # res / 4
            ResidualBlock(256, 256, downsample_factor=1),  # res / 4
            UpBlock(256, 64, scale_factor=2, upscaling_mode="bilinear"),  # res / 2
            ResidualBlock(64, 64, downsample_factor=1),  # res / 2
            UpBlock(64, 64, scale_factor=2, upscaling_mode="bilinear"),  # res
            ResidualBlock(64, 64, downsample_factor=1),  # res
            nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # res
        ]
        self.residuals = nn.Sequential(*residual_blocks)

    def forward(self, observations: torch.Tensor) -> DictWrapper[str, torch.Tensor]:
        """
        Computes the mask and the bounding box corresponding to each observation

        :param observations: (bs, 3 * observations_stacking, height, width) tensor
        :return: (bs, 1, height, width) tensor of masks
                 (bs, 4) tensor of bounding boxes
        """

        x = self.conv1(observations)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)

        x = self.residuals(x)

        mask = torch.sigmoid(x)

        return DictWrapper(
            masks=mask)


def build_masking_network(config: Configuration, convert_to_sequence: bool = False) -> nn.Module:
    backbone = MaskingNetwork(
        in_channels=config["in_channels"])

    if convert_to_sequence:
        return SequenceConverter(backbone=backbone)
    else:
        return backbone
