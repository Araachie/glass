from argparse import ArgumentParser, Namespace as ArgsNamespace
import torch

from dataset.dataset_splitter import DatasetSplitter
from dataset.transforms import TransformsGenerator
from dataset.video_dataset import VideoDataset
from evaluation.evaluator import Evaluator
from evaluation.evaluation_dataset_builder import builder
from model.model import Model

from utils.configuration import Configuration
from utils.logger import Logger


def parse_args() -> ArgsNamespace:
    parser = ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True, help="Name of the current run.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--step", type=int, default=None, help="Step to build the evaluation dataset for.")
    parser.add_argument("--random-seed", type=int, default=1543, help="Random seed.")

    return parser.parse_args()


def main():
    args = parse_args()
    config = Configuration(args.config)
    run_name = args.run_name
    step = args.step

    # Count gpus
    num_gpus = torch.cuda.device_count()
    assert num_gpus == 1, "Run this on 1 gpu only"
    device = torch.device("cuda:0")

    # Initialize logger
    extended_run_name = "{}_run-{}".format(config["name"], run_name)
    logger = Logger(
        project="global_local_motion",
        run_name=extended_run_name + "-evaluation",
        use_wandb=False,
        config=config,
        rank=0)

    # Setup dataset splits
    logger.info("Building datasets")

    dataset_splits = DatasetSplitter.generate_splits(config)
    transformations = TransformsGenerator.get_final_transforms(config)

    key = "test"
    path, batching_config, split = dataset_splits[key]
    transform = transformations[key]

    dataset = VideoDataset(
        path=path,
        batching_config=batching_config,
        final_transform=transform,
        allowed_videos=split,
        offset=config["data"]["offset"],
        max_num_videos=config["data"]["max_num_videos"],
        logger=logger)

    # Setup trainer
    logger.info("Instantiating evaluator object")
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=num_gpus,
        rank=0)
    evaluator = Evaluator(
        rank=0,
        run_name=extended_run_name,
        config=config["evaluation"],
        dataset=dataset,
        sampler=sampler,
        num_gpus=num_gpus,
        device=device)

    # Setup model and distribute across gpus
    logger.info("Building the model and distributing it across gpus")
    model = Model(config=config.model)
    model.cuda()
    if (num_gpus > 1) and len(list(model.parameters())) != 0:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[0],
            broadcast_buffers=False,
            find_unused_parameters=True)

    # Load checkpoint
    assert step is not None
    if step == -1:
        logger.info("Loading the latest checkpoint")
        evaluator.load_checkpoint(model)
    else:
        logger.info(f"Loading the checkpoint at step {step}")
        evaluator.load_checkpoint(model, f"step_{step}.pth")

    # Build dataset builder
    logger.info("Building the generated dataset")
    dataset_builder = builder(config=config["evaluation"], dataset=dataset)
    dataset_builder.build(model)


if __name__ == "__main__":
    main()
