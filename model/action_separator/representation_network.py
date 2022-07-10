import torch
import torch.nn as nn
import torch.nn.functional as F


from model.layers import ResidualBlock
from model.layers.utils import SequenceConverter

from utils.configuration import Configuration


class RepresentationNetwork(nn.Module):
    """
    Model that encodes an observation into a state

    """

    def __init__(self, in_channels: int, out_channels: int):
        super(RepresentationNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        residual_blocks = [
            ResidualBlock(16, 16, downsample_factor=1),  # res / 2
            ResidualBlock(16, 32, downsample_factor=2),  # res / 4
            ResidualBlock(32, 32, downsample_factor=1),  # res / 4
            ResidualBlock(32, 64, downsample_factor=2),  # res / 8
            ResidualBlock(64, 64, downsample_factor=1),  # res / 8
            ResidualBlock(64, out_channels + 1, downsample_factor=1),  # res / 8
        ]
        self.residuals = nn.Sequential(*residual_blocks)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Computes the state corresponding to each observation

        :param observations: (bs, 3 * observations_stacking, height, width) tensor
        :return: (bs, states_features, states_height, states_width) tensor of states
        """

        x = self.conv1(observations)
        x = F.avg_pool2d(x, 2)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)

        x = self.residuals(x)

        state = x

        return state


def build_representation_network(config: Configuration, convert_to_sequence: bool = False) -> nn.Module:
    backbone = RepresentationNetwork(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"])

    if convert_to_sequence:
        return SequenceConverter(backbone=backbone)
    else:
        return backbone
