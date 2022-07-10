import torch
import torch.nn as nn

from model.layers import SameBlock

from utils.configuration import Configuration
from utils.tensor_folder import TensorFolder


class MotionNetwork(nn.Module):
    def __init__(self, config: Configuration, output: str):
        super(MotionNetwork, self).__init__()

        self.config = config

        assert output in ["shift", "affine"]
        self.output_size = 2 if output == "shift" else 4

        self.motion_predictor = nn.Sequential(
            SameBlock(in_planes=6, out_planes=16, downsample_factor=2),  # res / 2
            SameBlock(in_planes=16, out_planes=32, downsample_factor=2),  # res / 4
            SameBlock(in_planes=32, out_planes=32),  # res / 4
            SameBlock(in_planes=32, out_planes=64, downsample_factor=2),  # res / 8
            SameBlock(in_planes=64, out_planes=128),  # res / 8
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, self.output_size))

        # Initialize the weights so that the output is initially a small shift
        self.motion_predictor[-1].weight.data.zero_()
        if self.output_size == 4:
            initial_affine = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
            initial_affine[2:] = 0.0001 * torch.randn(2, dtype=torch.float32)

            self.motion_predictor[-1].bias.data.copy_(initial_affine)
        else:
            initial_shift = 0.0001 * torch.randn(2, dtype=torch.float32)

            self.motion_predictor[-1].bias.data.copy_(initial_shift)

    def forward(self, observation_pairs: torch.Tensor) -> torch.Tensor:
        """
        Predicts the motion params

        :param observation_pairs: (bs, num_observations, 6, height, width)
        :return:
        """

        num_observations = observation_pairs.size(1)

        flat_observations_pairs = TensorFolder.flatten(observation_pairs)
        flat_motion_params = self.motion_predictor(flat_observations_pairs)
        folded_motion_params = TensorFolder.fold(flat_motion_params, num_observations)

        return folded_motion_params
