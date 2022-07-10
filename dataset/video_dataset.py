import os
from typing import Set, List, Dict

from torch.utils.data import Dataset

from dataset.video import Video
from dataset.batching import BatchElement

from utils.logger import Logger


class VideoDataset(Dataset):
    """
    Dataset of video objects
    Expects a directory where each children directory represents a Video object on disk
    """

    def __init__(
            self,
            path,
            batching_config: Dict,
            final_transform,
            allowed_videos=None,
            offset: int = None,
            max_num_videos: int = None,
            logger: Logger = None):
        """
        Initializes the dataset with the videos in a directory

        :param path: path to the root of the dataset
        :param batching_config: Dict with the batching parameters to use for sampling
        :param final_transform: transformation to apply to each frame
        :param allowed_videos: set of video names allowed to be part of the dataset.
                               if not None only videos in this set are included in the dataset
        :param offset: skip offset number of video from the beginning of the dataset
        :param max_num_videos: limits the number of videos loaded
        :param logger: logger object for logging
        """

        self.logger = logger

        if not os.path.isdir(path):
            raise Exception(f"Dataset directory '{path}' is not a directory")

        self.batching_config = batching_config

        # number of frames that compose each observation
        self.observations_stacking = batching_config['observation_stacking']
        # how many frames to skip between each observation
        self.skip_frames = batching_config['skip_frames']
        self.final_transform = final_transform

        # Reads the videos in the root
        self.all_videos = self.read_all_videos(path, allowed_videos, offset, max_num_videos)

        # These attributes will be defined later
        self.available_samples_list = None
        self.total_available_samples = None

        self.observations_count = None
        # number of observations to include in each dataset sample
        self.set_observations_count(batching_config['observations_count'])

    def set_observations_count(self, observations_count: int):
        """
        Changes the number of observations in each future returned sequence

        :param observations_count: Number of observations in each future returned sequence
        :return:
        """

        # Perform changes only if the parameter differs
        if self.observations_count is None or self.observations_count != observations_count:
            self.observations_count = observations_count

            self.available_samples_list = self.compute_available_samples_per_video()
            self.total_available_samples = sum(self.available_samples_list)

    def read_all_videos(
            self,
            path: str,
            allowed_videos: Set[str],
            offset: int = None,
            max_num_videos: int = None) -> List[Video]:
        """
        Reads all the allowed videos in the specified path

        :param path: path where videos are stored
        :param allowed_videos: set of video names allowed to be part of the dataset
                               if None all videos are included in the dataset
        :param offset: start reading from this offset
        :param max_num_videos: max number of videos to read
        :return:
        """

        all_videos = []
        contents = sorted(list(os.listdir(path)))

        # Allow everything if no limitations are specified
        if allowed_videos is None:
            allowed_videos = contents

        if offset is None:
            offset = 0
        if max_num_videos is None:
            max_num_videos = len(contents)
        for current_file in contents[offset:offset + max_num_videos]:
            current_file_path = os.path.join(path, current_file)
            if os.path.isdir(current_file_path) and current_file in allowed_videos:
                self.logger.info(f"Loading video at '{current_file_path}'")
                current_video = Video()
                current_video.load(current_file_path)
                all_videos.append(current_video)

        return all_videos

    def compute_available_samples_per_video(self) -> List[int]:
        """
        Computes how many samples can be drawn from the video sequences

        :return: list with an integer for each video representing how many samples can be drawn
        """

        available_samples = []

        # Number of frames in the original video each sample will span
        sample_block_size = self.observations_count + (self.observations_count - 1) * self.skip_frames

        for current_video in self.all_videos:
            frames_count = current_video.get_frames_count()
            current_samples = frames_count - sample_block_size + 1
            available_samples.append(current_samples)

        return available_samples

    def __len__(self):
        return self.total_available_samples

    def __getitem__(self, index) -> BatchElement:

        if index >= self.total_available_samples:
            raise Exception(f"Requested sample at index {index} is out of range")

        video_index = 0
        video_initial_frame = 0

        # Searches the video and the frame index in that video where to start extracting the sequence
        passed_samples = 0
        for search_index, current_available_samples in enumerate(self.available_samples_list):
            if passed_samples + current_available_samples > index:
                video_index = search_index
                video_initial_frame = index - passed_samples
                break
            passed_samples += current_available_samples

        current_video = self.all_videos[video_index]
        observation_indexes = []
        for i in range(self.observations_count):
            observation_indexes.append(video_initial_frame + i * (self.skip_frames + 1))

        # The minimum frame for which the preceding would not be part of the video
        min_frame = video_initial_frame % (self.skip_frames + 1)
        all_frames_indexes = [
            [max(current_observation_index - i * (self.skip_frames + 1), min_frame)
                for i in range(self.observations_stacking)]
            for current_observation_index in observation_indexes]
        all_frames = [
            [current_video.get_frame_at(index) for index in current_observation_stack]
            for current_observation_stack in all_frames_indexes]

        all_masks = [current_video.get_mask_at(current_index) for current_index in observation_indexes]

        all_global_actions = [current_video.global_actions[current_index] for current_index in observation_indexes]
        all_global_shifts = [current_video.global_shifts[current_index] for current_index in observation_indexes]
        all_local_actions = [current_video.local_actions[current_index] for current_index in observation_indexes]
        all_bg_shifts = [current_video.bg_shifts[current_index] for current_index in observation_indexes]

        return BatchElement(
            all_frames,
            all_masks,
            all_global_actions,
            all_global_shifts,
            all_local_actions,
            all_bg_shifts,
            current_video,
            video_initial_frame,
            self.final_transform)
