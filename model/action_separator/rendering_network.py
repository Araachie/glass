import torch
import torch.nn as nn

from typing import Any

from model.layers import FinalBlock, ResidualBlock, UpBlock
from model.layers.utils import SequenceConverter

from utils.configuration import Configuration
from utils.dict_wrapper import DictWrapper


class RenderingNetwork(nn.Module):
    """
    Model that reconstructs the frame associated to a hidden state
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(RenderingNetwork, self).__init__()

        bottleneck_block_list = [

        ]

        upsample_block_list = [
            nn.Sequential(UpBlock(in_channels, 128, scale_factor=2, upscaling_mode="bilinear"),  # res * 4
                          ResidualBlock(128, 128, downsample_factor=1)),
            nn.Sequential(UpBlock(128, 64, scale_factor=2, upscaling_mode="bilinear"),  # res * 8
                          ResidualBlock(64, 64, downsample_factor=1)),
            UpBlock(64, 32, scale_factor=2, upscaling_mode="bilinear"),  # res * 16
        ]

        final_block_list = [
            FinalBlock(128, out_channels, kernel_size=3, padding=1),
            FinalBlock(64, out_channels, kernel_size=3, padding=1),
            FinalBlock(32, out_channels + 1, kernel_size=7, padding=3)
        ]

        self.bottleneck_blocks = nn.Sequential(*bottleneck_block_list)
        self.upsample_blocks = nn.ModuleList(upsample_block_list)
        self.final_blocks = nn.ModuleList(final_block_list)

        if len(upsample_block_list) != len(final_block_list):
            raise Exception("Rendering network specifies a number of upsampling blocks"
                            "that differs from the number of final blocks")

    def forward(self, x: torch.Tensor) -> DictWrapper[str, Any]:
        """
        Computes the frames corresponding to each state at multiple resolutions

        :param x: (bs, representation_shape) tensor
        :return: [(bs, 3, height/2^i, width/2^i) for i in range(num_upsample_blocks)]
        """

        current_features = self.bottleneck_blocks(x)

        reconstructed_observations = []
        for upsample_block, final_block in zip(self.upsample_blocks, self.final_blocks):
            # Upsample the features
            current_features = upsample_block(current_features)
            # Transform them in the corresponding resolution image
            current_reconstructed_observation = final_block(current_features)
            reconstructed_observations.append(current_reconstructed_observation)

        # Inverts from high res to low res
        reconstructed_observations = list(reversed(reconstructed_observations))
        masks = 0.5 * (reconstructed_observations[0][:, [-1]] + 1)
        reconstructed_observations[0] = reconstructed_observations[0][:, :-1]
        return DictWrapper(
            reconstructed_observations=reconstructed_observations,
            masks=masks)


def build_rendering_network(config: Configuration, convert_to_sequence: bool = False) -> nn.Module:
    backbone = RenderingNetwork(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"])
    if convert_to_sequence:
        return SequenceConverter(backbone=backbone)
    else:
        return backbone
