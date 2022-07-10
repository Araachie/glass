from typing import Tuple

import torch
import torch.nn as nn

from model.layers import ConvLSTM, SameBlock, UpBlock, ModulatedConvBlock, channelwise_concat
from utils.configuration import Configuration


class ConvDynamicsNetwork(nn.Module):
    """
    Model that predicts the future state given the current state

    """

    def __init__(
            self,
            state_size: int,
            hidden_state_size: int,
            state_resolution: Tuple[int, ...],
            action_space_dimension: int,
            noise_size: int,
            modulated_conv: bool = True,
            joint_input: bool = False):
        super(ConvDynamicsNetwork, self).__init__()

        self.state_size = state_size
        self.hidden_state_size = hidden_state_size
        self.state_resolution = state_resolution
        self.noise_size = noise_size
        self.modulated_conv = modulated_conv
        self.joint_input = joint_input
        self.auxiliary_input_size = 2 * state_size + noise_size if joint_input else state_size + noise_size

        assert float(not modulated_conv) * float(joint_input) == 0

        # The recurrent layers used by the model
        self.recurrent_layers = [
            ConvLSTM(
                self.state_size + self.auxiliary_input_size,
                self.hidden_state_size,
                self.state_resolution),
            ConvLSTM(
                2 * self.hidden_state_size + self.auxiliary_input_size,
                2 * self.hidden_state_size,
                (self.state_resolution[0] // 2, self.state_resolution[1] // 2)),
            ConvLSTM(
                self.hidden_state_size + self.auxiliary_input_size,
                self.hidden_state_size,
                self.state_resolution)
        ]

        # Blocks with the recurrent layers and their normalization + activation
        self.recurrent_layers_blocks = nn.ModuleList([
            nn.Sequential(self.recurrent_layers[0], nn.BatchNorm2d(self.hidden_state_size)),
            nn.Sequential(self.recurrent_layers[1], nn.BatchNorm2d(2 * self.hidden_state_size)),
            nn.Sequential(self.recurrent_layers[2], nn.BatchNorm2d(self.hidden_state_size))
        ])

        if not modulated_conv:
            self.non_recurrent_blocks = nn.ModuleList([
                SameBlock(
                    self.hidden_state_size + self.auxiliary_input_size,
                    2 * self.hidden_state_size,
                    downsample_factor=2),
                UpBlock(
                    2 * self.hidden_state_size + self.auxiliary_input_size,
                    self.hidden_state_size,
                    upscaling_mode="bilinear",
                    late_upscaling=True),
                SameBlock(
                    self.hidden_state_size + self.auxiliary_input_size,
                    self.hidden_state_size,
                    downsample_factor=1)
            ])
        elif joint_input:
            self.non_recurrent_blocks = nn.ModuleList([
                ModulatedConvBlock(
                    self.hidden_state_size + noise_size,
                    2 * self.hidden_state_size,
                    2 * state_size,
                    downsample_factor=2),
                UpBlock(
                    2 * self.hidden_state_size + self.auxiliary_input_size,
                    self.hidden_state_size,
                    upscaling_mode="bilinear",
                    late_upscaling=True),
                ModulatedConvBlock(
                    self.hidden_state_size + noise_size,
                    self.hidden_state_size,
                    2 * state_size,
                    downsample_factor=1)
            ])
        else:
            self.non_recurrent_blocks = nn.ModuleList([
                ModulatedConvBlock(
                    self.hidden_state_size + noise_size,
                    2 * self.hidden_state_size,
                    state_size,
                    downsample_factor=2),
                UpBlock(
                    2 * self.hidden_state_size + self.auxiliary_input_size,
                    self.hidden_state_size,
                    upscaling_mode="bilinear",
                    late_upscaling=True),
                ModulatedConvBlock(
                    self.hidden_state_size + noise_size,
                    self.hidden_state_size,
                    state_size,
                    downsample_factor=1)
            ])

        self.la_projection = nn.Linear(action_space_dimension, state_size)
        self.ga_projection = nn.Linear(2, state_size)

    def reinit_memory(self):
        """
        Initializes the state of the recurrent cells
        """

        # Initializes memory
        for current_layer in self.recurrent_layers:
            current_layer.reinit_memory()

    def forward(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            global_actions: torch.Tensor,
            randomize: bool = False) -> torch.Tensor:
        """
        Computes the successor states given the selected actions and noise
        Current states are maintained implicitly and are reset through reinit_memory
        reinit_memory must have been called at least once before forward

        :param states: (bs, states_features, states_height, states_width) tensor
        :param actions: (bs, action_space_dimension) tensor
        :param global_actions: (bs, 2) tensor
        :param randomize: whether to use random noise

        :return: (bs, hidden_state_size) tensor with the successor states
        """

        batch_size = states.size(0)

        if self.noise_size > 0:
            if randomize:
                random_noise = torch.randn([batch_size, self.noise_size], device=states.device)
            else:
                random_noise = torch.zeros([batch_size, self.noise_size], device=states.device)

        p_actions = self.la_projection(actions)
        p_global_actions = self.ga_projection(global_actions)

        # Passes the input tensors through each block, concatenating the auxiliary inputs at each step
        for i, (rec_block, non_rec_block) in enumerate(zip(self.recurrent_layers_blocks, self.non_recurrent_blocks)):
            if self.joint_input:
                action_cond = torch.cat([p_actions, p_global_actions], dim=1)
            else:
                action_cond = p_global_actions if i == 0 else p_actions

            # Pass through the recurrent block
            input_states = [states, action_cond]
            if self.noise_size > 0:
                input_states.append(random_noise)
            states = rec_block(input_states)

            # Pass through the non-recurrent block
            input_states = [states]
            if not self.modulated_conv or i == 1:
                input_states.append(action_cond)
            if self.noise_size > 0:
                input_states.append(random_noise)
            if not self.modulated_conv or i == 1:
                states = non_rec_block(
                    channelwise_concat(input_states, non_rec_block.in_planes))
            else:
                states = non_rec_block(
                    channelwise_concat(input_states, non_rec_block.in_planes),
                    action_cond)

        return states


def build_conv_dynamics_network(config: Configuration) -> nn.Module:
    return ConvDynamicsNetwork(
        state_size=config["state_size"],
        hidden_state_size=config["hidden_state_size"],
        state_resolution=config["state_resolution"],
        action_space_dimension=config["action_space_dimension"],
        noise_size=config["noise_size"],
        modulated_conv=config["modulated_conv"],
        joint_input=config["joint_input"])
