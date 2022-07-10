from argparse import ArgumentParser, Namespace as ArgsNamespace

from evaluation.dataset_evaluator_tennis import evaluator as tennis_evaluator
from evaluation.dataset_evaluator import evaluator as other_evaluator

from dataset.dataset_splitter import DatasetSplitter
from dataset.transforms import TransformsGenerator
from dataset.video_dataset import VideoDataset

from utils.configuration import Configuration
from utils.logger import Logger


def parse_args() -> ArgsNamespace:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--random-seed", type=int, default=1543, help="Random seed.")

    return parser.parse_args()


def main():
    args = parse_args()
    config = Configuration(args.config)

    # Initialize logger
    extended_run_name = "{}_run-dataset_evaluation".format(config["name"])
    logger = Logger(
        project="glass",
        run_name=extended_run_name,
        use_wandb=False,
        config=config,
        rank=0)

    # Setup dataset splits
    logger.info("Building datasets")

    dataset_splits = DatasetSplitter.generate_splits(config)
    transformations = TransformsGenerator.get_final_transforms(config)

    key = "test"
    path, batching_config, split = dataset_splits[key]

    reference_transform, generated_transform = TransformsGenerator.get_evaluation_transforms(config)

    reference_dataset = VideoDataset(
        path=path,
        batching_config=batching_config,
        final_transform=reference_transform,
        allowed_videos=split,
        offset=None,
        max_num_videos=config["data"]["max_num_videos"],
        logger=logger)
    generated_dataset = VideoDataset(
        path=config["evaluation"]["evaluation_dataset_directory"],
        batching_config=batching_config,
        final_transform=generated_transform,
        allowed_videos=None,
        offset=None,
        max_num_videos=None,
        logger=logger)

    # Evaluate the dataset
    logger.info("Evaluating the dataset")
    if config["name"].startswith("tennis"):
        evaluator = tennis_evaluator
    else:
        evaluator = other_evaluator
    dataset_evaluator = evaluator(config["evaluation"], logger, reference_dataset, generated_dataset)
    results = dataset_evaluator.compute_metrics()
    print(results)


if __name__ == "__main__":
    main()
