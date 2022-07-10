# This model is able to segment foregrounds

from typing import Any

import torch
import torch.nn as nn

from model.action_separator import ActionSeparator
from model.inpainting_network import InpaintingNetwork
from model.masking_network import build_masking_network
from model.motion_network import MotionNetwork
from model.utils import spatial_transform
from utils.configuration import Configuration
from utils.dict_wrapper import DictWrapper
from utils.tensor_folder import TensorFolder


class Model(nn.Module):
    """
    A class that handles the whole computational pipeline

    """

    def __init__(self, config: Configuration):
        super(Model, self).__init__()

        self.config = config

        self.local_action_separator = ActionSeparator(
            config=config["action_separator"])

        self.masking_network = build_masking_network(
            config=config["masking_network"],
            convert_to_sequence=True)

        self.foreground_transform_predictor = MotionNetwork(
            config=config["motion_network"],
            output="shift")

        self.background_shift_predictor = MotionNetwork(
            config=config["motion_network"],
            output="shift")

        self.inpainting_network = InpaintingNetwork(
            config=config["inpainting_network"])

    def forward(self, observations: torch.Tensor, path: str, **kwargs: Any) -> DictWrapper[str, Any]:
        """
        Performs a forward pass

        :param observations: (bs, num_observations, 3 * observation_stacking, height, width)
        :param path: which path to forward through
        :param kwargs: additional args
        :return:
        """

        assert path in ["main"]

        if path == "main":
            return self.forward_main_path(observations, **kwargs)
        else:
            raise RuntimeError(f"Model does not have a path f{path}")

    def calculate_masks(self, observations: torch.Tensor) -> DictWrapper[str, Any]:
        """
        Predict the masks, the bounding boxes and crop observations

        :param observations: (bs, num_observations, 3 * observation_stacking, height, width)
        :return:
        """

        # Compute representation
        masking_results = self.masking_network(observations)
        masks = masking_results.masks

        return DictWrapper(
            masks=masks)

    def shift_backgrounds(
            self,
            backgrounds: torch.Tensor,
            shifts: torch.Tensor = None,
            interpolation_mode: str = "bilinear") -> DictWrapper[str, Any]:
        """
        Predicts the shift for the backgrounds and shifts them

        :param backgrounds: (bs, num_observations, 3, height, width)
        :param shifts: (bs, num_observations|num_observations - 1, 2)
        :param interpolation_mode: ["nearest"|"bilinear"|etc]
        :return:
        """

        batch_size = backgrounds.size(0)
        num_observations = backgrounds.size(1)
        num_channels = backgrounds.size(2)
        height = backgrounds.size(3)
        width = backgrounds.size(4)

        # Predict shifts
        predecessor_bg = backgrounds[:, :-1]
        if shifts is None:
            successor_bg = backgrounds[:, 1:]
            bg_pairs = torch.cat([predecessor_bg, successor_bg], dim=2)
            predicted_shifts = self.background_shift_predictor(bg_pairs)
        else:
            predicted_shifts = shifts
        flat_shifts = TensorFolder.flatten(predicted_shifts)
        if predicted_shifts.size(1) == backgrounds.size(1):
            predecessor_bg = backgrounds
            num_observations += 1

        # Prepare transforms with identity scales
        identity_scale = torch.Tensor([1.0, 1.0]).to(flat_shifts.device)
        identity_scale = identity_scale.unsqueeze(0).expand(batch_size * (num_observations - 1), -1)
        flat_transforms = torch.cat([identity_scale, flat_shifts], dim=1)

        # Shift backgrounds
        flat_shifted_backgrounds = spatial_transform(
            TensorFolder.flatten(predecessor_bg),
            flat_transforms,
            [batch_size * (num_observations - 1), num_channels, height, width],
            inverse=True,
            padding_mode="border",
            mode=interpolation_mode)
        folded_shifted_backgrounds = TensorFolder.fold(flat_shifted_backgrounds, num_observations - 1)

        return DictWrapper(
            shifted_backgrounds=folded_shifted_backgrounds,
            predicted_shifts=predicted_shifts)

    def transform_foregrounds(self, foregrounds: torch.Tensor, masks: torch.Tensor) -> DictWrapper[str, Any]:
        """
        Predicts the affine transform for the foregrounds and applies it

        :param foregrounds: (bs, num_observations, 3, height, width)
        :param masks: (bs, num_observations, 1, height, width)
        :return:
        """

        batch_size = foregrounds.size(0)
        num_observations = foregrounds.size(1)
        num_channels = foregrounds.size(2)
        height = foregrounds.size(3)
        width = foregrounds.size(4)

        # Predict transforms
        predecessor_fg = foregrounds[:, :-1]
        successor_fg = foregrounds[:, 1:]
        fg_pairs = torch.cat([predecessor_fg, successor_fg], dim=2)
        shifts = self.foreground_transform_predictor(fg_pairs)
        flat_shifts = TensorFolder.flatten(shifts)

        # Prepare transforms with identity scales
        identity_scale = torch.Tensor([1.0, 1.0]).to(flat_shifts.device)
        identity_scale = identity_scale.unsqueeze(0).expand(batch_size * (num_observations - 1), -1)
        flat_transforms = torch.cat([identity_scale, flat_shifts], dim=1)

        # Apply transforms to the foregrounds
        flat_transformed_foregrounds = spatial_transform(
            TensorFolder.flatten(predecessor_fg),
            flat_transforms,
            [batch_size * (num_observations - 1), num_channels, height, width],
            inverse=True,
            padding_mode="zeros",
            mode="bilinear")
        folded_transformed_foregrounds = TensorFolder.fold(flat_transformed_foregrounds, num_observations - 1)

        # Apply transforms to the masks
        flat_transformed_masks = spatial_transform(
            TensorFolder.flatten(masks[:, :-1]),
            flat_transforms,
            [batch_size * (num_observations - 1), 1, height, width],
            inverse=True,
            padding_mode="zeros",
            mode="bilinear")
        folded_transformed_masks = TensorFolder.fold(flat_transformed_masks, num_observations - 1)

        return DictWrapper(
            foreground_shifts=shifts,
            transformed_foregrounds=folded_transformed_foregrounds,
            transformed_masks=folded_transformed_masks)

    @staticmethod
    def paste(
            crops: torch.Tensor,
            bounding_boxes: torch.Tensor,
            transforms: torch.Tensor,
            height: int,
            width: int) -> DictWrapper[str, Any]:
        """

        :param crops: (bs, num_observations, 3, height, width)
        :param bounding_boxes: (bs, num_observations, 4)
        :param transforms: (bs, num_observations, 2) shifts
        :param height: height of the full frame
        :param width: width of the full frame
        :return:
        """

        batch_size = crops.size(0)
        num_observations = crops.size(1)
        num_channels = crops.size(2)

        flat_pasted_crops = spatial_transform(
            TensorFolder.flatten(crops),
            TensorFolder.flatten(bounding_boxes),
            [batch_size * num_observations, num_channels, height, width],
            inverse=True)

        # Prepare transforms with identity scales
        flat_transforms = TensorFolder.flatten(transforms)
        identity_scale = torch.Tensor([1.0, 1.0]).to(flat_transforms.device)
        identity_scale = identity_scale.unsqueeze(0).expand(batch_size * num_observations, -1)
        flat_transforms = torch.cat([identity_scale, flat_transforms], dim=1)

        flat_pasted_crops = spatial_transform(
            flat_pasted_crops,
            flat_transforms,
            [batch_size * num_observations, num_channels, height, width],
            inverse=True,
            padding_mode="zeros",
            mode="bilinear")

        pasted_crops = TensorFolder.fold(flat_pasted_crops, num_observations)

        return DictWrapper(
            pasted_crops=pasted_crops)

    def forward_main_path(
            self,
            observations: torch.Tensor,
            num_ground_truth_observations: int,
            gumbel_temperature: float,
            output_generated: bool = False) -> DictWrapper[str, Any]:
        """
        Performs a forward pass through the main pipeline

        :param observations: (bs, num_observations, 3 * observation_stacking, height, width)
        :param num_ground_truth_observations: Number of ground truth frames to use
        :param gumbel_temperature: Gumbel temperature to use
        :param output_generated: Whether to output the generated sequences or not
        :return:
        """

        observations = observations[:, :, :3]

        # Compute masks, bounding_boxes and crops
        masks = self.calculate_masks(observations).masks

        # Calculate backgrounds and foregrounds
        foregrounds = masks * observations
        backgrounds = (1 - masks) * observations

        # Inpaint backgrounds
        inpainting_results = self.inpainting_network(observations, masks, randomize_masks=False, combine=True)
        inpainted_backgrounds = inpainting_results.inpainted_backgrounds
        patched_backgrounds = inpainting_results.patched_backgrounds

        # Shift backgrounds
        background_shift_results = self.shift_backgrounds(inpainted_backgrounds)
        shifted_backgrounds = background_shift_results.shifted_backgrounds
        background_shifts = background_shift_results.predicted_shifts

        # Transform foregrounds
        foreground_transform_results = self.transform_foregrounds(foregrounds, masks)
        transformed_foregrounds = foreground_transform_results.transformed_foregrounds
        transformed_masks = foreground_transform_results.transformed_masks
        global_actions = foreground_transform_results.foreground_shifts

        # Perform local action separation
        action_separation_results = self.local_action_separator(
            foregrounds.detach(),
            transformed_foregrounds.detach(),
            global_actions.detach(),
            gumbel_temperature=gumbel_temperature,
            num_ground_truth_observations=num_ground_truth_observations)
        local_states = action_separation_results.states
        reconstructed_foregrounds = action_separation_results.reconstructed_observations
        reconstructed_masks = action_separation_results.reconstructed_masks
        reconstructed_local_states = action_separation_results.reconstructed_states
        local_action_vq_loss = action_separation_results.action_vq_loss
        local_action_ids = action_separation_results.action_ids

        # Combine layers
        reconstructed_observations = \
            transformed_masks * transformed_foregrounds + (1 - transformed_masks) * shifted_backgrounds
        generated_observations = reconstructed_observations

        if output_generated:
            with torch.no_grad():
                inpainted_first_background = self.inpainting_network(
                    observations[:, [0]],
                    masks[:, [0]],
                    randomize_masks=False,
                    combine=True,
                    enlarge_masks=True).inpainted_backgrounds
                stacked_initial_backgrounds = inpainted_first_background.expand(-1, backgrounds.size(1) - 1, -1, -1, -1)
                accumulated_shifts = torch.cumsum(background_shifts, dim=1)
                shifted_initial_backgrounds = self.shift_backgrounds(
                    stacked_initial_backgrounds,
                    accumulated_shifts,
                    interpolation_mode="nearest").shifted_backgrounds
                generated_observations = \
                    reconstructed_masks * reconstructed_foregrounds[0] + \
                    (1 - reconstructed_masks) * shifted_initial_backgrounds

        return DictWrapper(
            # Input
            observations=observations,

            # Masking
            masks=masks,
            foregrounds=foregrounds,
            backgrounds=backgrounds,

            # Background shifting
            patched_backgrounds=patched_backgrounds,
            inpainted_backgrounds=inpainted_backgrounds,
            shifted_backgrounds=shifted_backgrounds,
            background_shifts=background_shifts,

            # Foreground transform
            transformed_foregrounds=transformed_foregrounds,
            global_actions=global_actions,

            # Local action separator
            local_states=local_states,
            reconstructed_foregrounds=reconstructed_foregrounds,
            reconstructed_masks=reconstructed_masks,
            reconstructed_local_states=reconstructed_local_states,
            local_action_vq_loss=local_action_vq_loss,
            local_action_ids=local_action_ids,

            # Output
            reconstructed_observations=reconstructed_observations,
            generated_observations=generated_observations
        )
