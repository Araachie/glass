import os
from typing import List
from tqdm import tqdm

import numpy as np
import torch
from PIL import Image

from torch.utils.data import DataLoader

from dataset.batching import single_batch_elements_collate_fn
from dataset.video import Video
from dataset.video_dataset import VideoDataset


class EvaluationDatasetBuilder:
    """
    Helper class for model evaluation
    """

    def __init__(self, config, dataset: VideoDataset):

        self.config = config
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.config["batching"]["batch_size"],
            shuffle=False,
            collate_fn=single_batch_elements_collate_fn,
            num_workers=self.config["batching"]["num_workers"],
            pin_memory=True)

        self.output_path = self.config["evaluation_dataset_directory"]
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

    def build(self, model):
        """
        Builds the evaluation dataset

        :param model: The model to use for building the evaluation dataset
        :return:
        """

        all_videos = []

        # Setup loading bar
        validation_gen = tqdm(self.dataloader, desc="Batches", disable=False, leave=False)

        with torch.no_grad():
            for idx, batch in enumerate(validation_gen):

                # Performs inference
                observations, masks, global_actions, global_shifts, local_actions, bg_shifts, sprite_ids = \
                    batch.to_tuple()

                model_outputs = model(
                    observations,
                    path="main",
                    num_ground_truth_observations=observations.size(1) - 1,
                    gumbel_temperature=self.config["gumbel_temperature"],
                    output_generated=True)
                reconstructed_observations = model_outputs.generated_observations

                # Pads the reconstructed observations with the first ground truth image
                reconstructed_observations = torch.cat([observations[:, 0:1, 0:3], reconstructed_observations], dim=1)
                # Normalizes the range of the observations
                reconstructed_observations = self.check_and_normalize_range(reconstructed_observations)

                # Converts to numpy
                reconstructed_observations = reconstructed_observations.cpu().numpy()
                # Moves the channels as the last dimension
                reconstructed_observations = np.moveaxis(reconstructed_observations, 2, -1)

                # Builds the video objects for the current batch
                current_videos = self.predictions_to_videos(
                    reconstructed_observations)
                all_videos.extend(current_videos)

        # Creates the dataset
        self.create_dataset(self.output_path, all_videos)

    @staticmethod
    def predictions_to_videos(images: np.ndarray) -> List[Video]:
        """

        :param images: (bs, observations_count, height, width, channels) tensor
        :return:
        """

        images = (images * 255).astype(np.uint8)

        batch_size, sequence_length, height, width, channels = images.shape

        all_videos = []
        # Transforms a sequence at a time into a video
        for sequence_idx in range(batch_size):
            current_images = images[sequence_idx]
            current_images = [Image.fromarray(current_image) for current_image in current_images]

            # Creates the current video
            current_video = Video()
            current_video.add_frames(current_images)
            all_videos.append(current_video)

        return all_videos

    @staticmethod
    def create_dataset(path, videos: List[Video], extension="png"):
        """
        Creates a dataset with the given video sequences

        :param path: path where to save the dataset
        :param videos: list of the videos to save
        :param extension: frame extension
        :return:
        """

        for idx, video in enumerate(videos):
            current_path = os.path.join(path, f"{idx:05d}")
            video.save(current_path, extension)

    @staticmethod
    def check_and_normalize_range(observations: torch.Tensor) -> torch.Tensor:
        """
        If the range of the observations is in [-1, 1] instead of [0, 1] it normalizes it
        :param observations: arbitrarily shaped tensor to normalize
        :return: the input tensor normalized in [0, 1]
        """

        minimum_value = torch.min(observations).item()

        # Check range and normalize
        if minimum_value < 0:
            observations = (observations + 1) / 2
        return observations


def builder(config, dataset: VideoDataset):
    return EvaluationDatasetBuilder(config, dataset)
