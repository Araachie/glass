from argparse import Namespace as ArgsNamespace
from typing import Any, Dict

import numpy as np

import torch

from dataset import DatasetSplitter, TransformsGenerator, VideoDataset
from model import Model
from evaluation.evaluator import Evaluator
from utils.logger import Logger
from utils.configuration import Configuration
from utils.constants import MAIN_PROCESS


def setup_evaluation_arguments(args: ArgsNamespace) -> Dict[str, Any]:
    evaluation_args = dict()

    # Load config file
    evaluation_args["config"] = Configuration(args.config)

    # Other args
    evaluation_args["run_name"] = args.run_name
    evaluation_args["step"] = args.step
    evaluation_args["random_seed"] = args.random_seed
    evaluation_args["num_gpus"] = args.num_gpus
    evaluation_args["use_wandb"] = args.wandb

    return evaluation_args


def launch_evaluation(
        rank: int,
        config: Configuration,
        run_name: str,
        step: int,
        cudnn_benchmark: bool = True,
        random_seed: int = None,
        num_gpus: int = 1,
        use_wandb: bool = False):
    # Initialize some stuff
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    is_main_process = rank == MAIN_PROCESS
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Initialize logger
    extended_run_name = "{}_run-{}".format(config["name"], run_name)
    logger = Logger(
        project="glass",
        run_name=extended_run_name + "-evaluation",
        use_wandb=use_wandb,
        config=config,
        rank=rank)

    # Setup dataset splits
    logger.info("Building datasets")

    dataset_splits = DatasetSplitter.generate_splits(config)
    transformations = TransformsGenerator.get_final_transforms(config)

    key = "validation"
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

    # Setup evaluator
    logger.info("Instantiating evaluator object")
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=num_gpus,
        rank=rank)
    evaluator = Evaluator(
        rank=rank,
        run_name=extended_run_name,
        config=config["evaluation"],
        dataset=dataset,
        sampler=sampler,
        num_gpus=num_gpus,
        device=device)

    # Setup model and distribute across gpus
    logger.info("Building the model and distributing it across gpus")
    model = Model(config=config.model)
    model.to(device)
    if (num_gpus > 1) and len(list(model.parameters())) != 0:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
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

    # Launch the training loop
    logger.info("Launching evaluation")
    evaluator.evaluate(model=model, logger=logger)
