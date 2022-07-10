from typing import List, Tuple

import torch
from torchvision.transforms import transforms as T
from PIL import Image

from dataset.video import Video


class BatchElement:
    def __init__(
            self,
            observations: List[Tuple[Image.Image]],
            masks: List[Image.Image],
            global_actions: List,
            global_shifts: List,
            local_actions: List,
            bg_shifts: List,
            video: Video,
            initial_frame_index: int,
            transforms: T):
        """
        Constructs a batch element

        :param observations: list of observations_count tuples with observations_stacking frames each from the most recent to the oldest
        :param masks: list of masks
        :param global_actions: list of observations_count actions
        :param global_shifts: list of observations_count shifts
        :param local_actions: list of observations_count actions
        :param bg_shifts: list of observations_count background shifts
        :param video: the original video object
        :param initial_frame_index: the index in the original video of the frame corresponding to the first observation
        :param transforms: transform to apply to each frame in the observations. Must return torch tensors
        """

        self.observations_count = len(observations)
        self.observations_stacking = len(observations[0])

        if len(global_actions) != self.observations_count or len(local_actions) != self.observations_count:
            raise Exception("Missing elements in the current batch")

        self.global_actions = global_actions
        self.global_shifts = global_shifts
        self.local_actions = local_actions
        self.bg_shifts = bg_shifts
        self.video = video
        self.initial_frame_index = initial_frame_index
        self.transforms = transforms
        self.mask_transforms = T.Compose(transforms.transforms[:2])

        self.observations = []
        for current_observation in observations:
            transformed_observation = [self.transforms(frame) for frame in current_observation]
            self.observations.append(transformed_observation)

        self.masks = [self.mask_transforms(mask) for mask in masks]


class Batch:
    def __init__(
            self,
            observations: torch.Tensor,
            masks: torch.Tensor,
            global_actions: torch.Tensor,
            global_shifts: torch.Tensor,
            local_actions: torch.Tensor,
            bg_shifts: torch.Tensor,
            videos: List[Video],
            initial_frames: List[int]):
        """

        :param observations: (bs, observations_count, 3 * observations_stacking, h, w) tensor with observed images
        :param masks: (bs, observations_count, 1, h, w) tensor with masks
        :param global_actions: (bs, observations_count) tensor with observed actions
        :param global_shifts: (bs, observations_count, 2) tensor with observed shifts
        :param local_actions: (bs, observations_count) tensor with observed actions
        :param bg_shifts: (bs, observations_count) tensor with observed background shifts
        :param videos: list of original bs videos
        :param initial_frames: list of integers representing indexes in the original videos corresponding to the first frame
        """

        self.size = global_actions.size(1)

        self.observations = observations
        self.masks = masks
        self.global_actions = global_actions
        self.global_shifts = global_shifts
        self.local_actions = local_actions
        self.bg_shifts = bg_shifts
        self.video = videos
        self.sprite_ids = torch.Tensor([v.sprite_id for v in videos]).to(torch.int64)
        self.initial_frames = initial_frames

    def to_cuda(self):
        """
        Transfers tensors to the gpu

        :return:
        """
        self.observations = self.observations.cuda()
        self.masks = self.masks.cuda()
        self.global_actions = self.global_actions.cuda()
        self.global_shifts = self.global_shifts.cuda()
        self.local_actions = self.local_actions.cuda()
        self.bg_shifts = self.bg_shifts.cuda()
        self.sprite_ids = self.sprite_ids.cuda()

    def to_tuple(self, cuda=True) -> Tuple:
        """
        Converts the batch to an input tuple

        :param cuda If True transfers the tensors to the gpu
        :return: (observations, actions, rewards, dones) tuple
        """

        if cuda:
            self.to_cuda()

        return (
            self.observations,
            self.masks,
            self.global_actions,
            self.global_shifts,
            self.local_actions,
            self.bg_shifts,
            self.sprite_ids
        )

    def pin_memory(self):
        self.observations.pin_memory()
        self.masks.pin_memory()
        self.global_actions.pin_memory()
        self.global_shifts.pin_memory()
        self.local_actions.pin_memory()
        self.bg_shifts.pin_memory()
        self.sprite_ids.pin_memory()

        return self


def single_batch_elements_collate_fn(batch: List[BatchElement]) -> Batch:
    """
    Creates a batch starting from single batch elements

    :param batch: List of batch elements
    :return: Batch representing the passed batch elements
    """

    observations_tensor = torch.stack(
        [torch.stack([torch.cat(current_stack) for current_stack in current_element.observations], dim=0)
         for current_element in batch], dim=0)
    masks_tensor = torch.stack(
        [torch.stack([current_mask for current_mask in current_element.masks], dim=0)
         for current_element in batch], dim=0)
    global_actions_tensor = torch.stack(
        [torch.tensor(current_element.global_actions, dtype=torch.int) for current_element in batch], dim=0)
    global_shifts_tensor = torch.stack(
        [torch.stack([torch.tensor([x, y], dtype=torch.float32) for x, y in current_element.global_shifts], dim=0)
         for current_element in batch], dim=0)
    local_actions_tensor = torch.stack(
        [torch.tensor(current_element.local_actions, dtype=torch.int) for current_element in batch], dim=0)
    bg_shifts_tensor = torch.stack(
        [torch.stack([torch.tensor([x, y], dtype=torch.float32) for x, y in current_element.bg_shifts], dim=0)
         for current_element in batch], dim=0)
    videos = [current_element.video for current_element in batch]
    initial_frames = [current_element.initial_frame_index for current_element in batch]

    return Batch(
        observations_tensor,
        masks_tensor,
        global_actions_tensor,
        global_shifts_tensor,
        local_actions_tensor,
        bg_shifts_tensor,
        videos,
        initial_frames)
