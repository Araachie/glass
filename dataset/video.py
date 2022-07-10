from typing import List, Tuple

import glob
from PIL import Image
import os
import pickle


class Video:
    """
     A video from the dataset
     Frames are loaded from the disk on demand only
    """

    def __init__(self):
        # These attributes will be defined later
        self.frames = None
        self.extension = None
        self.frames_path = None
        self.frames_count = None

        # This is only used if the dataset is W-Sprites
        # Is inferred from the path name to the video frames
        self.sprite_id = -1

        self.has_masks = False
        self.masks = None

        self.global_actions = None
        self.global_shifts = None
        self.local_actions = None
        self.bg_shifts = None

    def add_frames(
            self,
            frames: List[Image.Image]):
        """
        Adds the contents to the video

        :param frames: list of all video frames
        :return:
        """

        self.frames = frames

        self.extension = None
        self.frames_path = None
        self.frames_count = len(self.frames)

    @staticmethod
    def _index_to_filename(idx):
        return f'{idx:05}'

    def check_none_coherency(self, sequence):
        """
        Checks that the sequence either has all values set to None or to a not None value
        Raises an exception if the sequence does not satisfy the criteria

        :param sequence: the sequence to check
        :return:
        """

        has_none = False
        has_not_none = False

        for element in sequence:
            if element is None:
                has_none = True
            else:
                has_not_none = True
            if has_none and has_not_none:
                raise Exception(f"Video dataset at {self.frames_path} metadata error:"
                                f"both None and not None data are present")

    def check_metadata_and_set_defaults(self):
        """
        Checks the medata and sets default values if None are present
        :return:
        """

        # Checks coherency of None values in the metadata
        self.check_none_coherency(self.global_actions)
        self.check_none_coherency(self.local_actions)

        if self.global_actions[0] is None:
            self.global_actions = [0] * len(self.global_actions)
        if self.global_shifts[0] is None:
            self.global_shifts = [(0, 0)] * len(self.global_shifts)
        if self.local_actions[0] is None:
            self.local_actions = [0] * len(self.local_actions)
        if self.bg_shifts[0] is None:
            self.bg_shifts = [(0, 0)] * len(self.bg_shifts)

    def load(self, path):
        if not os.path.isdir(path):
            raise Exception(f"Cannot load video: '{path}' is not a directory")

        # Frames are not load into memory
        self.frames_path = path

        # Calculate sprite_id if the dataset is W-Sprites
        if path.split("/")[-1].startswith("sprite"):
            self.sprite_id = int(path.split("/")[-1].split("_")[1])

        # Discover extension of videos
        padded_index = self._index_to_filename(0)
        results = glob.glob(os.path.join(path, f'{padded_index}.*'))
        if len(results) != 1:
            raise Exception("Could not find first video frame")
        extension = results[0].split(".")[-1]
        self.extension = extension

        # Count the number of frames
        results = glob.glob(os.path.join(path, f'*.{self.extension}'))
        self.frames_count = len(results)

        # Load actions
        if os.path.exists(os.path.join(path, "glactions.pkl")):
            with open(os.path.join(path, "glactions.pkl"), "rb") as f:
                actions = pickle.load(f)

                self.global_actions = actions["global"]
                self.global_shifts = actions["shifts"]
                self.local_actions = actions["local"]
                self.bg_shifts = actions["bg_shifts"]
        elif os.path.exists(os.path.join(path, "actions.pkl")):
            with open(os.path.join(path, "actions.pkl"), "rb") as f:
                actions = pickle.load(f)

                self.global_actions = actions
                self.global_shifts = [None] * len(actions)
                self.local_actions = actions
                self.bg_shifts = [None] * len(actions)
        else:
            self.global_actions = [None] * self.frames_count
            self.global_shifts = [None] * self.frames_count
            self.local_actions = [None] * self.frames_count
            self.bg_shifts = [None] * self.frames_count

        if self.frames_count != len(self.local_actions) or self.frames_count != len(self.global_actions):
            raise Exception("Read data have inconsistent number of frames")

        # Set default values in the metadata if needed
        self.check_metadata_and_set_defaults()

        # Check if the dataset has masks
        if os.path.exists(os.path.join(path, "masks")):
            results = glob.glob(os.path.join(path, "masks", f'*.{self.extension}'))
            assert len(results) == self.frames_count
            self.has_masks = True

    def get_frames_count(self) -> int:
        if self.frames_count is None:
            raise Exception("Video has not been initialized. Did you forget to call load()?")

        return self.frames_count

    def get_frame_at(self, idx: int) -> Image:
        """
        Returns the frame corresponding to the specified index

        :param idx: index of the frame to retrieve in [0, frames_count-1]
        :return: The frame at the specified index
        """
        if self.frames_count is None:
            raise Exception("Video has not been initialized. Did you forget to call load()?")
        if idx < 0 or idx >= self.frames_count:
            raise Exception(f"Index {idx} is out of range")

        # If frames are load into memory
        if self.frames is not None:
            return self.frames[idx]

        padded_index = self._index_to_filename(idx)
        filename = os.path.join(self.frames_path, f'{padded_index}.{self.extension}')
        image = Image.open(filename)
        image = self.remove_transparency(image)
        return image

    def get_mask_at(self, idx: int) -> Image:
        """
        Returns the mask corresponding to the specified index

        :param idx: index of the mask to retrieve in [0, frames_count-1]
        :return: The mask at the specified index
        """
        if self.frames_count is None:
            raise Exception("Video has not been initialized. Did you forget to call load()?")
        if idx < 0 or idx >= self.frames_count:
            raise Exception(f"Index {idx} is out of range")

        # If frames are load into memory
        if self.masks is not None:
            return self.masks[idx]

        if not self.has_masks:
            return Image.new("L", (1, 1))

        padded_index = self._index_to_filename(idx)
        filename = os.path.join(self.frames_path, "masks", f'{padded_index}.{self.extension}')
        image = Image.open(filename)
        image = self.remove_transparency(image)
        return image

    @staticmethod
    def remove_transparency(image, bg_colour=(255, 255, 255)):
        # Only process if image has transparency (http://stackoverflow.com/a/1963146)
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):

            # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
            alpha = image.convert('RGBA').split()[-1]

            # Create a new background image of our matt color.
            # Must be RGBA because paste requires both images have the same format
            # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
            bg = Image.new("RGBA", image.size, bg_colour + (255,))
            bg.paste(image, mask=alpha)
            bg = bg.convert("RGB")
            return bg
        else:
            return image

    def save(self, path: str, extension="png"):
        if self.frames_count is None:
            raise Exception("Video has not been initialized. Did you forget to call add_frames()?")
        if os.path.isdir(path):
            raise Exception(f"A directory at '{path}' already exists")

        # Creates the directory
        os.mkdir(path)

        # Save all frames
        for idx, frame in enumerate(self.frames):
            padded_index = self._index_to_filename(idx)
            filename = os.path.join(path, f'{padded_index}.{extension}')
            frame.save(filename)
