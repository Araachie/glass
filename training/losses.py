import sys
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import Vgg19
from utils.tensor_folder import TensorFolder


class ReconstructionLoss(nn.Module):
    """
    L1 loss
    """

    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    @staticmethod
    def forward(
            observations: torch.Tensor,
            reconstructed_observations: torch.Tensor,
            weights: torch.Tensor = None) -> torch.Tensor:
        """
        Computes L1 loss between observations and reconstructions

        :param observations: (bs, num_observations, 3, height, width)
        :param reconstructed_observations: (bs, num_observations, 3, height, width)
        :param weights: (bs, num_observations, 1, height, width)
        :return: (1,)
        """

        # Make sure the spatial dims are the same
        assert observations.size(3) == reconstructed_observations.size(3)
        assert observations.size(4) == reconstructed_observations.size(4)

        loss = torch.abs(observations - reconstructed_observations)
        if weights is not None:
            if weights.size(3) != observations.size(3) or weights.size(4) != observations.size(4):
                weights = MultiscaleLoss.resize(weights, (observations.size(3), observations.size(4)))
            loss *= weights

        return loss.mean()


class PerceptualLoss(nn.Module):
    """
    L1 loss in VGG feature space
    """

    def __init__(self):
        super(PerceptualLoss, self).__init__()

        self.vgg = Vgg19()

    def forward(self, observations: torch.Tensor, reconstructed_observations: torch.Tensor) -> torch.Tensor:
        """
        Computes the perceptual loss between observations and reconstructions

        :param observations: (bs, num_observations, 3, height, width)
        :param reconstructed_observations: (bs, num_observations, 3, height, width)
        :return: (1,)
        """

        # Make sure the spatial dims are the same
        assert observations.size(3) == reconstructed_observations.size(3)
        assert observations.size(4) == reconstructed_observations.size(4)

        ground_truth_features = self.vgg(TensorFolder.flatten(observations.detach()))
        reconstructed_features = self.vgg(TensorFolder.flatten(reconstructed_observations))

        loss = 0
        for cur_ground_truth_features, cur_reconstructed_features in zip(
                ground_truth_features, reconstructed_features):
            loss += torch.abs(cur_ground_truth_features - cur_reconstructed_features).mean()

        return loss


class MultiscaleLoss(nn.Module):
    """
    Wraps a loss into a sequence computation pipeline
    """

    def __init__(self, loss: nn.Module):
        super(MultiscaleLoss, self).__init__()

        self.loss = loss

    @staticmethod
    def resize(x: torch.Tensor,  target_resolution: Tuple[int, int]) -> torch.Tensor:
        """
        Resizes x to the target resolution

        :param x: (bs, num_observations, observation_stacking, Height, Width)
        :param target_resolution: (height, width)
        :return: resized_x: (bs, num_observations, 3, height, width)
        """

        num_observations = x.size(1)
        flat_x = TensorFolder.flatten(x)
        flat_resized_x = F.interpolate(
            flat_x, size=target_resolution, mode="bilinear", align_corners=False)
        resized_x = TensorFolder.fold(flat_resized_x, num_observations)

        return resized_x

    def resize_and_align(self, observations: torch.Tensor, target_resolution: Tuple[int, int]) -> torch.Tensor:
        """
        Resizes observations to the target resolution

        :param observations: (bs, num_observations, 3 * observation_stacking, Height, Width)
        :param target_resolution: (height, width)
        :return: resized and aligned: (bs, num_observations, 3, height, width)
        """

        # Drop the first observation and leave only one frame from the rest
        observations = observations[:, 1:, :3]

        return self.resize(observations, target_resolution)

    def forward(
            self,
            observations: torch.Tensor,
            reconstructed_observations: List[torch.Tensor],
            **kwargs) -> torch.Tensor:
        """
        Computes L1 loss across multiple scales

        :param observations: (bs, num_observations, 3 * observation_stacking, Height, Width)
        :param reconstructed_observations: k * [(bs, num_observations - 1, 3, height, width)]
        :return:
        """

        loss = 0
        for current_reconstruction in reconstructed_observations:
            target_height = current_reconstruction.size(3)
            target_width = current_reconstruction.size(4)

            resized_observations = self.resize_and_align(observations, (target_height, target_width))

            loss += self.loss(resized_observations, current_reconstruction, **kwargs)

        return loss / len(reconstructed_observations)


class BackgroundShiftLoss(nn.Module):
    """
    Loss matching the shifted backgrounds
    """

    def __init__(self):
        super(BackgroundShiftLoss, self).__init__()

    @staticmethod
    def forward(
            backgrounds: torch.Tensor,
            shifted_backgrounds: torch.Tensor) -> torch.Tensor:
        """

        :param backgrounds: (bs, num_observations, 3, height, width)
        :param shifted_backgrounds: (bs, num_observations - 1, 3, height, width)
        :return:
        """

        loss = torch.abs(backgrounds[:, 1:] - shifted_backgrounds)

        return loss.mean()


class ForegroundTransformLoss(nn.Module):
    """
    Loss matching the transformed foregrounds
    """

    def __init__(self):
        super(ForegroundTransformLoss, self).__init__()

    @staticmethod
    def forward(
            foregrounds: torch.Tensor,
            transformed_foregrounds: torch.Tensor) -> torch.Tensor:
        """

        :param foregrounds: (bs, num_observations, 3, height, width)
        :param transformed_foregrounds: (bs, num_observations - 1, 3, height, width)
        :return:
        """

        loss = torch.pow(foregrounds[:, 1:] - transformed_foregrounds, 2)

        return loss.mean()


class BinaryMaskLoss(nn.Module):
    """
    Loss imposing binary mask
    """

    def __init__(self):
        super(BinaryMaskLoss, self).__init__()

    @staticmethod
    def forward(masks: torch.Tensor) -> torch.Tensor:
        """

        :param masks: (bs, num_observations, 1, h, w)
        :return:
        """

        loss = torch.minimum(masks, 1 - masks)

        return loss.mean()


class MaskSizeLoss(nn.Module):
    """
    Loss on the size of the mask
    """

    def __init__(self, size: float):
        super(MaskSizeLoss, self).__init__()

        self.size = size

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        """

        :param masks: (bs, num_observations, 1, h, w)
        :return:
        """

        loss = torch.abs(masks.mean([-1, -2]) - self.size)

        return loss.mean()
