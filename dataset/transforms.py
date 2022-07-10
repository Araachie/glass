from typing import Dict, List, Tuple

from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms

from utils.configuration import Configuration


class ResizeAndCropTransform(nn.Module):
    def __init__(self, target_crop: List[int], target_size: Tuple[int]):
        super(ResizeAndCropTransform, self).__init__()

        self.target_crop = target_crop
        self.target_size = target_size

    def forward(self, image: Image):
        if self.target_crop is not None:
            image = image.crop(self.target_crop)
        if image.size != tuple(self.target_size):
            image = image.resize(self.target_size, Image.BILINEAR)

        return image


class TransformsGenerator:

    @staticmethod
    def check_and_resize(target_crop: List[int], target_size: Tuple[int]):
        """
        Creates a function that transforms input PIL images to the target size

        :param target_crop: [left_index, upper_index, right_index, lower_index] list representing the crop region
        :param target_size: (width, height) touple representing the target height and width
        :return: function that transforms a PIL image to the target size
        """

        # Creates the transformation function
        def transform(image: Image):
            if target_crop is not None:
                image = image.crop(target_crop)
            if image.size != tuple(target_size):
                image = image.resize(target_size, Image.BILINEAR)

            return image

        return transform

    @staticmethod
    def to_float_tensor(tensor):
        return tensor / 1.0

    @staticmethod
    def get_final_transforms(config: Configuration) -> Dict[str, transforms.Compose]:
        """
        Obtains the transformations to use for training and evaluation

        :param config: The configuration file
        :return:
        """

        transform = transforms.Compose([
            ResizeAndCropTransform(
                config["data"]["crop"],
                config["data"]["input_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        return {
            "train": transform,
            "validation": transform,
            "test": transform,
        }

    @staticmethod
    def get_evaluation_transforms(config: Configuration) -> Tuple:
        """
        Obtains the transformations to use for the evaluation scripts

        :param config: The evaluation configuration file
        :return: reference_transformation, generated transformation to use for the reference and the generated datasets
        """

        reference_resize_transform = ResizeAndCropTransform(
            config["data"]["crop"],
            config["data"]["input_size"])
        generated_resize_transform = ResizeAndCropTransform(
            config["data"]["crop"],
            config["data"]["input_size"])

        # Do not normalize data for evaluation
        reference_transform = transforms.Compose([reference_resize_transform,
                                                  transforms.ToTensor(),
                                                  TransformsGenerator.to_float_tensor])
        generated_transform = transforms.Compose([generated_resize_transform,
                                                  transforms.ToTensor(),
                                                  TransformsGenerator.to_float_tensor])

        return reference_transform, generated_transform
