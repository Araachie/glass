from typing import List

import torch
import torch.nn as nn

from utils.tensor_folder import TensorFolder


def make_2d_tensor(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Transforms a 1d tensor into a 2d tensor of specified dimensions

    :param tensor: (bs, features) tensor
    :param height: int
    :param width: int
    :return: (bs, features, height, width) tensor with repeated features along the spatial dimensions
    """

    tensor = tensor.unsqueeze(dim=-1).unsqueeze(dim=-1)  # Adds two final dimensions for broadcast
    tensor = tensor.repeat((1, 1, height, width))  # Repeats along the spatial dimensions

    return tensor


def channelwise_concat(inputs: List[torch.Tensor], in_planes: int) -> torch.Tensor:
    """
    Concatenates all inputs tensors channelwise

    :param inputs: [(bs, features_i, height, width) / (bs, features_i)] list of tensor which feature dimensions sum to in_planes
    :param in_planes: int
    :return: (bs, in_planes, height, width)
    """

    # Infers the target spatial dimensions
    height = 0
    width = 0
    for current_tensor in inputs:
        if len(current_tensor.size()) == 4:
            height = current_tensor.size(2)
            width = current_tensor.size(3)
            break
    if height == 0 or width == 0:
        raise Exception("No tensor in the inputs has a spatial dimension."
                        "Ensure at least one tensor represents a tensor with spatial dimensions")

    # Expands tensors to spatial dimensions
    expanded_tensors = []
    for current_tensor in inputs:
        if len(current_tensor.size()) == 4:
            expanded_tensors.append(current_tensor)
        elif len(current_tensor.size()) == 2:
            expanded_tensors.append(make_2d_tensor(current_tensor, height, width))
        else:
            raise Exception("Expected tensors with 2 or 4 dimensions")

    # Concatenates tensors channelwise
    concatenated_tensor = torch.cat(expanded_tensors, dim=1)
    total_features = concatenated_tensor.size(1)
    if total_features != in_planes:
        raise Exception(
            f"The input tensors features sum to {total_features}, but layer takes {in_planes} features as input")

    return concatenated_tensor


class SequenceConverter(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(SequenceConverter, self).__init__()

        self.backbone = backbone

    @staticmethod
    def convert(x, n):
        if isinstance(x, list):
            return [TensorFolder.fold(e, n) for e in x]
        return TensorFolder.fold(x, n)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        observations_count = sequences.size(1)

        x = TensorFolder.flatten(sequences)
        x = self.backbone(x)

        if isinstance(x, dict):
            for k, v in x.items():
                x[k] = self.convert(v, observations_count)
        else:
            x = self.convert(x, observations_count)

        return x
