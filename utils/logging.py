from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid

from utils.tensor_folder import TensorFolder


@torch.no_grad()
def make_observations_grid(
        observations: torch.Tensor,
        reconstructed_observations: torch.Tensor,
        other: List[torch.Tensor],
        num_sequences: int) -> torch.Tensor:
    """
    Formats the observations into a grid.

    :param observations: [bs, num_observations, 3 * observation_stacking, Height, Width]
    :param reconstructed_observations: [bs, num_observations - 1, 3, height, width]
    :param other: List of [bs, num_observations, 3, height, width]
    :param num_sequences: Number of sequences to log
    :return: The grid of observations for logging.
    """

    _, _, _, h, w = reconstructed_observations.shape

    def pad(x):
        if x.size(1) == observations.size(1):
            return x.cpu()[:num_sequences]
        elif x.size(1) == observations.size(1) - 1:
            num_sequences_pad = min(x.size(0), num_sequences)
            return torch.cat([
                torch.zeros([num_sequences_pad, 1, 3, h, w]),
                x.cpu()[:num_sequences]
            ], dim=1)
        else:
            raise RuntimeError("wrong number of observations in reconstructed_observations")

    def resize(x):
        if x.size(3) == h and x.size(4) == w:
            return x.cpu()[:num_sequences, :, :3]
        else:
            n = x.size(1)
            y = F.interpolate(
                TensorFolder.flatten(x.cpu()[:num_sequences, :, :3]),
                size=(h, w),
                mode="bilinear",
                align_corners=False)
            return TensorFolder.fold(y, n)

    def add_channels(x):
        if x.size(2) == 1:
            return x.expand(-1, -1, 3, -1, -1)
        else:
            return x

    # Pad reconstructed observations
    padded_reconstructed_observations = pad(reconstructed_observations)

    # Resize the observations to the reconstructed size
    resized_observations = resize(observations)

    # Pad and resize other to the reconstructed size
    resized_other = [to_image(pad(resize(add_channels(x)))) for x in other]

    # Put the observations one next to another
    stacked_observations = torch.stack([
        resized_observations,
        padded_reconstructed_observations] +
        resized_other,
        dim=1)
    flat_observations = TensorFolder.flatten(
        TensorFolder.flatten(stacked_observations)
    )

    return make_grid(flat_observations, nrow=flat_observations.size(0) // ((2 + len(other)) * num_sequences))


@torch.no_grad()
def to_video(x: torch.Tensor) -> np.array:
    return (((x + 1) / 2).detach().cpu().numpy() * 255).astype(np.uint8)


@torch.no_grad()
def to_image(x: torch.Tensor) -> torch.Tensor:
    return (((x + 1) / 2).detach().cpu() * 255).to(torch.uint8)


@torch.no_grad()
def calculate_local_action_matrix(model: nn.Module, first_frames: torch.Tensor, num_frames: int) -> np.array:
    """
    Generates action matrix in a form
    [f_1 a_1, f_1 a_2, ... f_1 a_k]
    [f_2 a_1, f_2 a_2, ... f_2 a_k]
    ...
    [f_m a_1, f_m a_2, ... f_m a_k]

    :param model: Model
    :param first_frames: (bs, 1, 3 * observation_stacking, height, width)
    :param num_frames: Number of frames to generate
    :return:
    """

    batch_size, _, _, height, width = first_frames.shape

    is_distributed = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    dmodel = model.module if is_distributed else model
    dmodel = dmodel.local_action_separator

    num_actions = dmodel.action_network.num_actions

    columns = []
    for action in range(num_actions):
        dmodel.start_inference()

        cur_observations = first_frames
        frames = [cur_observations[:, :, :3]]
        for i in range(num_frames - 1):
            next_frames, next_observations, _ = dmodel.generate_next(cur_observations, action, [0.0, 0.0])
            frames.append(next_frames)
            cur_observations = next_observations

        cur_column = torch.cat(frames, dim=1)  # (bs, num_frames, 3, height, width)
        cur_column = cur_column.permute(1, 2, 0, 3, 4)
        cur_column = cur_column.reshape(num_frames, 3, batch_size * height, width)
        columns.append(cur_column)

    action_matrix = torch.stack(columns, dim=3)
    action_matrix = action_matrix.reshape(num_frames, 3, batch_size * height, num_actions * width)

    return to_video(action_matrix)


@torch.no_grad()
def calculate_global_action_matrix(
        model: nn.Module,
        first_frames: torch.Tensor,
        num_frames: int,
        local_action: int = 0) -> np.array:
    """
    Generates action matrix in a form
    [f_1 a_1, f_1 a_2, ... f_1 a_k]
    [f_2 a_1, f_2 a_2, ... f_2 a_k]
    ...
    [f_m a_1, f_m a_2, ... f_m a_k]

    :param model: Model
    :param first_frames: (bs, 1, 3 * observation_stacking, height, width)
    :param num_frames: Number of frames to generate
    :param local_action: Local action to use in generation
    :return:
    """

    batch_size, _, _, height, width = first_frames.shape

    is_distributed = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    dmodel = model.module if is_distributed else model
    dmodel = dmodel.local_action_separator

    global_actions = [
        [0.07, 0.0],
        [-0.07, 0.0],
        [0.0, 0.1],
        [0.0, -0.1],
        [0.0, 0.0],
    ]

    columns = []
    for global_action in global_actions:
        dmodel.start_inference()

        cur_observations = first_frames
        frames = [cur_observations[:, :, :3]]
        for i in range(num_frames - 1):
            next_frames, next_observations, _ = dmodel.generate_next(cur_observations, local_action, global_action)
            frames.append(next_frames)
            cur_observations = next_observations

        cur_column = torch.cat(frames, dim=1)  # (bs, num_frames, 3, height, width)
        cur_column = cur_column.permute(1, 2, 0, 3, 4)
        cur_column = cur_column.reshape(num_frames, 3, batch_size * height, width)
        columns.append(cur_column)

    action_matrix = torch.stack(columns, dim=3)
    action_matrix = action_matrix.reshape(num_frames, 3, batch_size * height, len(global_actions) * width)

    return to_video(action_matrix)
