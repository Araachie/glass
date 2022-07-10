from typing import List, Tuple

import torch
import torch.nn as nn

from model.layers.utils import channelwise_concat


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell
    """

    def __init__(self, in_planes: int, out_planes: int):
        """

        :param in_planes: Number of input channels
        :param out_planes: Number of output channels
        """

        super(ConvLSTMCell, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes

        self.input_gate = nn.Conv2d(
            in_planes + self.out_planes, out_planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.forget_gate = nn.Conv2d(
            in_planes + self.out_planes, out_planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.output_gate = nn.Conv2d(
            in_planes + self.out_planes, out_planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.cell_gate = nn.Conv2d(
            in_planes + self.out_planes, out_planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)

    def forward(
            self,
            inputs: List[torch.Tensor],
            hidden_states: torch.Tensor,
            hidden_cell_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the successor states given the inputs

        :param inputs: [(bs, features_i, height, width) / (bs, features_i)] list of tensor which feature dimensions sum to in_planes
        :param hidden_states: (bs, out_planes, height, width) tensor with hidden state
        :param hidden_cell_states: (bs, out_planes, height, width) tensor with hidden cell state

        :return: (bs, out_planes, height, width), (bs, out_planes, height, width) tensors with hidden_state and hidden_cell_state
        """

        inputs.append(hidden_states)  # Also hidden states must be convolved with the input
        concatenated_input = channelwise_concat(inputs, self.in_planes + self.out_planes)

        # Processes the gates
        i = torch.sigmoid(self.input_gate(concatenated_input))
        f = torch.sigmoid(self.forget_gate(concatenated_input))
        o = torch.sigmoid(self.output_gate(concatenated_input))
        c = torch.tanh(self.cell_gate(concatenated_input))

        # Computes successor states
        successor_hidden_cell_states = f * hidden_cell_states + i * c

        successor_hidden_state = o * torch.tanh(successor_hidden_cell_states)

        return successor_hidden_state, successor_hidden_cell_states
