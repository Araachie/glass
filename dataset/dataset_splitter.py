import os
from typing import Dict


class DatasetSplitter:
    """
    Helper class for dataset splitting
    """

    @staticmethod
    def generate_splits(config) -> Dict:
        """
        Computes the subsets of directory to include in the train, validation and test splits

        :param config: the configuration file
        :return: dictionary with a list of directories for each split
        """

        dataset_style = config["data"]["dataset_style"]
        training_batching = config["training"]["batching"]
        evaluation_batching = config["evaluation"]["batching"]

        # If the data lies in a single directory that needs to be split
        if dataset_style == "flat":
            directory_contents = list(sorted(os.listdir(config["data"]["data_root"])))
            dataset_splits = config["data"]["dataset_splits"]

            contents_length = len(directory_contents)
            num_train = int(contents_length * dataset_splits[0])
            num_val = int(contents_length * dataset_splits[1])

            path = config["data"]["data_root"]
            return {
                "train": (path, training_batching, directory_contents[:num_train]),
                "validation": (path, evaluation_batching, directory_contents[num_train:num_train + num_val]),
                "test": (path, evaluation_batching, directory_contents[num_train + num_val:])
            }
        elif dataset_style == "splitted":

            base_path = config["data"]["data_root"]
            return {
                "train": (os.path.join(base_path, "train"), training_batching, None),
                "validation": (os.path.join(base_path, "val"), evaluation_batching, None),
                "test": (os.path.join(base_path, "test"), evaluation_batching, None)
            }
        else:
            raise Exception(f"Unknown dataset style '{dataset_style}'")
