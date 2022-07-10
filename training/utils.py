import re

import torch
import torch.nn as nn


def calculate_motion_weights(
        observations: torch.Tensor,
        reconstructed_observations: torch.Tensor,
        weight_bias: float = 0.0) -> torch.Tensor:
    """
    Calculates motion weights

    :param observations: [bs, num_observations, 3 * observation_stacking, height, width]
    :param reconstructed_observations: [bs, num_observations|num_observations - 1, 3, height, width]
    :param weight_bias: A value to add to the calculated mask
    :return: [bs, num_observations|num_observations - 1, 3, height, width]
    """

    # No gradient must flow through the computation of the weight masks
    observations = observations.detach()
    reconstructed_observations = reconstructed_observations.detach()

    # For each observation extract only the current frame and not the past ones
    observations = observations[:, :, :3]

    sequence_length = observations.size(1)
    reconstructed_sequence_length = reconstructed_observations.size(1)

    # If the length of the sequences differ, use the first sequence frame
    # to fill the first missing position in the reconstructed
    if reconstructed_sequence_length != sequence_length:
        if reconstructed_sequence_length != sequence_length - 1:
            raise Exception(
                f"Received an input batch with sequence length {sequence_length},"
                f"but got a reconstructed batch of {reconstructed_sequence_length}")
        reconstructed_observations = torch.cat([observations[:, 0:1], reconstructed_observations], dim=1)

    # Ensure the sequences have the same length
    assert (sequence_length == reconstructed_observations.size(1))

    # Computes corresponding predecessor and successor observations
    successor_observations = observations[:, 1:]
    predecessor_observations = observations[:, :-1]
    successor_reconstructed_observations = reconstructed_observations[:, 1:]
    predecessor_reconstructed_observations = reconstructed_observations[:, :-1]

    weight_mask = torch.abs(successor_observations - predecessor_observations) + \
        torch.abs(successor_reconstructed_observations - predecessor_reconstructed_observations)

    # Sums the mask along the channel dimension
    assert (weight_mask.size(2) == 3)
    weight_mask = weight_mask.sum(dim=2, keepdim=True)

    # Adds bias to the weights
    weight_mask += weight_bias

    return weight_mask


def calculate_local_motion_weights(
        observations: torch.Tensor,
        shifted_observations: torch.Tensor,
        reconstructed_observations: torch.Tensor,
        weight_bias: float = 0.0) -> torch.Tensor:
    """
    Calculates motion weights

    :param observations: [bs, num_observations, 3 * observation_stacking, height, width]
    :param shifted_observations: [bs, num_observations - 1, 3, height, width]
    :param reconstructed_observations: [bs, num_observations - 1, 3, height, width]
    :param weight_bias: A value to add to the calculated mask
    :return: [bs, num_observations - 1, 3, height, width]
    """

    # No gradient must flow through the computation of the weight masks
    observations = observations.detach()
    shifted_observations = shifted_observations.detach()
    reconstructed_observations = reconstructed_observations.detach()

    # For each observation extract only the current frame and not the past ones starting from the 2nd frame
    observations = observations[:, 1:, :3]

    sequence_length = observations.size(1)
    shifted_sequence_length = shifted_observations.size(1)
    reconstructed_sequence_length = reconstructed_observations.size(1)

    # If the length of the sequences differ, raise an error
    assert shifted_sequence_length == sequence_length and sequence_length == reconstructed_sequence_length

    weight_mask = torch.abs(observations - shifted_observations) + \
        torch.abs(observations - reconstructed_observations)

    # Sums the mask along the channel dimension
    assert (weight_mask.size(2) == 3)
    weight_mask = weight_mask.sum(dim=2, keepdim=True)

    # Adds bias to the weights
    weight_mask += weight_bias

    return weight_mask


def _named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())


def check_ddp_consistency(module: nn.Module, ignore_regex: str = None):
    for name, tensor in _named_params_and_buffers(module):
        fullname = type(module).__name__ + '.' + name
        if ignore_regex is not None and re.fullmatch(ignore_regex, fullname):
            continue
        tensor = tensor.detach()
        if tensor.is_floating_point():
            tensor = torch.nan_to_num(tensor)
        other = tensor.clone()
        torch.distributed.broadcast(tensor=other, src=0)
        assert torch.all(tensor == other), fullname
