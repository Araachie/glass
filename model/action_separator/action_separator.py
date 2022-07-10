from typing import Any, Tuple

import torch
import torch.nn as nn

from model.action_separator.action_network import build_action_network
from model.action_separator.rendering_network import build_rendering_network
from model.action_separator.representation_network import build_representation_network
from model.action_separator.conv_dynamics_network import build_conv_dynamics_network
from utils.configuration import Configuration
from utils.dict_wrapper import DictWrapper
from utils.tensor_folder import TensorFolder


class ActionSeparator(nn.Module):
    """
    A class that handles the whole computational pipeline
    """

    def __init__(self, config: Configuration):
        super(ActionSeparator, self).__init__()

        self.config = config

        self.representation_network = build_representation_network(
            config["representation_network"],
            convert_to_sequence=True)

        feature_size = config["representation_network"]["out_channels"]
        compressed_feature_size = config["state_compressor"]["out_channels"]
        self.state_compressor = nn.Sequential(
            nn.Conv2d(feature_size, compressed_feature_size, (1, 1), (1, 1), 0)
        )

        self.action_network = build_action_network(
            config["action_network"])

        self.dynamics_network = build_conv_dynamics_network(
            config["dynamics_network"])

        self.rendering_network = build_rendering_network(
            config["rendering_network"],
            convert_to_sequence=True)

    def forward(
            self,
            observations: torch.Tensor,
            transformed_observations: torch.Tensor,
            global_actions: torch.Tensor,
            gumbel_temperature: float,
            num_ground_truth_observations: int) -> DictWrapper[str, Any]:
        """
        Performs a forward pass through the main pipeline

        :param observations: (bs, num_observations, 3 * observation_stacking, height, width)
        :param transformed_observations: (bs, num_observations - 1, 3 * observation_stacking, height, width)
        :param global_actions: (bs, num_observations - 1, 2)
        :param gumbel_temperature: TAU in gumbel softmax trick
        :param num_ground_truth_observations: Number of ground truth observations to use during training
        :return:
        """

        num_observations = observations.size(1)

        # Encode observations and split to states and attentions
        representation = self.representation_network(observations)
        states = representation[:, :, :-1]
        attentions = representation[:, :, [-1]]
        shifted_representation = self.representation_network(transformed_observations)
        shifted_states = shifted_representation[:, :, :-1]
        shifted_attentions = shifted_representation[:, :, [-1]]

        # Extract compressed state
        flat_states = TensorFolder.flatten(states)
        flat_compressed_states = self.state_compressor(flat_states)
        compressed_states = TensorFolder.fold(flat_compressed_states, num_observations)

        # Calculate actions
        action_network_results = self.action_network(
            shifted_states,
            shifted_attentions,
            states[:, 1:],
            attentions[:, 1:],
            gumbel_temperature)
        action_codes = action_network_results.action_codes
        action_vq_loss = action_network_results.vq_loss
        action_ids = action_network_results.action_ids

        # Initialize dynamics memory
        self.dynamics_network.reinit_memory()

        # Predict future states for ground truth images
        predicted_recurrent_states_from_gt = []
        for i in range(num_ground_truth_observations):
            current_compressed_state = compressed_states[:, i]
            current_action_code = action_codes[:, i]
            current_global_action = global_actions[:, i]

            next_recurrent_state = self.dynamics_network(
                current_compressed_state, current_action_code, current_global_action)
            predicted_recurrent_states_from_gt.append(next_recurrent_state)
        predicted_recurrent_states_from_gt = torch.stack(predicted_recurrent_states_from_gt, dim=1)

        # Decode frames
        rendering_results = self.rendering_network(predicted_recurrent_states_from_gt)
        reconstructed_observations = rendering_results.reconstructed_observations
        reconstructed_masks = rendering_results.masks

        # Start autoregressive part
        for i in range(num_ground_truth_observations, num_observations - 1):
            current_observation = self.get_next_observation(observations, reconstructed_observations[0])
            current_representation = self.representation_network(current_observation)
            current_state = current_representation[:, 0, :-1]
            current_compressed_state = self.state_compressor(current_state)
            current_action_code = action_codes[:, i]
            current_global_action = global_actions[:, i]

            next_recurrent_state = self.dynamics_network(
                current_compressed_state, current_action_code, current_global_action)
            rendering_results = self.rendering_network(next_recurrent_state.unsqueeze(1))
            next_frame = rendering_results.reconstructed_observations
            next_mask = rendering_results.masks

            for r, new_frame in enumerate(next_frame):
                reconstructed_observations[r] = torch.cat([reconstructed_observations[r], new_frame], dim=1)
            reconstructed_masks = torch.cat([reconstructed_masks, next_mask], dim=1)

        # Encode reconstructed frames
        reconstructed_stacked_observations = self.stack_observations(
            observations,
            reconstructed_observations[0])
        reconstructed_representation = self.representation_network(
            reconstructed_stacked_observations)
        reconstructed_states = reconstructed_representation[:, :, :-1]

        return DictWrapper(
            observations=observations,
            states=states,
            action_vq_loss=action_vq_loss,
            action_ids=action_ids,
            reconstructed_observations=reconstructed_observations,
            reconstructed_masks=reconstructed_masks,
            reconstructed_states=reconstructed_states)

    @staticmethod
    def get_next_observation(observations: torch.Tensor, reconstructed_observations: torch.Tensor) -> torch.Tensor:
        """
        Stacks previously reconstructed observations to form a new one that is going to be passed to the network

        :param observations: (bs, num_observations, 3 * observations_stacking, height, width)
        :param reconstructed_observations: (bs, k, 3, height, width)
        :return: next_observation (bs, 1, 3 * observation_stacking, height, width)
        """

        batch_size, num_reconstructed_observations, _, height, width = reconstructed_observations.shape

        assert observations.size(2) % 3 == 0
        observations_stacking = observations.size(2) // 3

        next_observation = observations[:, [num_reconstructed_observations]]

        reconstructed_observations_to_paste = reconstructed_observations[:, -observations_stacking:]
        reconstructed_observations_to_paste = torch.flip(reconstructed_observations_to_paste, dims=[1])
        num_reconstructed_observations_to_paste = reconstructed_observations_to_paste.size(1)
        reconstructed_observations_to_paste = reconstructed_observations_to_paste.reshape(
            batch_size,
            num_reconstructed_observations_to_paste * 3,
            height,
            width)
        reconstructed_observations_to_paste = reconstructed_observations_to_paste.unsqueeze(1)

        next_observation[:, :, :3 * num_reconstructed_observations_to_paste] = reconstructed_observations_to_paste

        return next_observation

    @staticmethod
    def stack_observations(observations: torch.Tensor, reconstructed_observations: torch.Tensor) -> torch.Tensor:
        """
        Stack reconstructed observations to pass them to the representation network

        :param observations: (bs, num_observations, 3 * observations_stacking, height, width)
        :param reconstructed_observations: (bs, num_observations|num_observations - 1, 3, height, width)
        :return: (bs, num_observations - 1, 3 * observations_stacking, height, width)
        """

        num_observations = observations.size(1)
        if reconstructed_observations.size(1) == num_observations:
            reconstructed_observations = reconstructed_observations[:, 1:]

        assert observations.size(2) % 3 == 0
        observations_stacking = observations.size(2) // 3

        if observations_stacking == 1:
            return reconstructed_observations

        stacked_reconstructed_observations = torch.cat([reconstructed_observations, observations[:, 1:, 3:]], dim=2)
        for i in range(min(num_observations - 1, observations_stacking)):
            if i != 0:
                stacked_reconstructed_observations[:, i:, 3 * i:3 * (i + 1)] = reconstructed_observations[:, :-i]
            else:
                stacked_reconstructed_observations[:, i:, 3 * i:3 * (i + 1)] = reconstructed_observations

        return stacked_reconstructed_observations

    def start_inference(self):
        """
        Reinitialize memory in the dynamics network
        """

        self.dynamics_network.reinit_memory()

    def generate_next(
            self,
            observations: torch.Tensor,
            action: int,
            global_action: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        :param observations: (bs, 1, 3 * observation_stacking, height, width)
        :param action: action id
        :param global_action: [x, y]
        :return: frame (bs, 1, 3, height, width),
                 observation (bs, 1, 3 * observation_stacking, height, width),
                 mask (bs, 1, 1, height, width)
        """

        batch_size = observations.size(0)

        # Calculate state
        representation = self.representation_network(observations)
        states = representation[:, 0, :-1]
        compressed_states = self.state_compressor(states)

        # Calculate action codes
        action_one_hots = torch.zeros(self.action_network.num_actions).to(observations.device)
        action_one_hots[action] = 1.0
        action_one_hots = action_one_hots.unsqueeze(0).expand(batch_size, -1)
        action_codes = torch.matmul(action_one_hots, self.action_network.action_codes_quantizer.embedding.weight)

        # Set up global actions
        global_actions_t = torch.tensor(global_action).unsqueeze(0).expand(batch_size, -1).to(observations.device)

        # Calculate next_state
        next_states = self.dynamics_network(compressed_states, action_codes, global_actions_t)

        # Generate next frame
        rendering_results = self.rendering_network(next_states.unsqueeze(1))
        next_frames = rendering_results.reconstructed_observations[0]

        # Calculate next observations
        next_observations = torch.cat([next_frames, observations[:, :, :-3]], dim=2)

        return next_frames, next_observations, rendering_results.masks
