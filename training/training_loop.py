from argparse import Namespace as ArgsNamespace
from tqdm import tqdm
from typing import Any, Dict

import numpy as np

import torch

from dataset import DatasetSplitter, TransformsGenerator, VideoDataset
from model import Model
from training.trainer import Trainer
from utils.logger import Logger
from utils.configuration import Configuration
from utils.constants import MAIN_PROCESS


def setup_training_arguments(args: ArgsNamespace) -> Dict[str, Any]:
    training_args = dict()

    # Load config file
    training_args["config"] = Configuration(args.config)

    # Other args
    training_args["run_name"] = args.run_name
    training_args["resume_step"] = args.resume_step
    training_args["random_seed"] = args.random_seed
    training_args["num_gpus"] = args.num_gpus
    training_args["use_wandb"] = args.wandb

    return training_args


def training_loop(
        rank: int,
        config: Configuration,
        run_name: str,
        resume_step: int,
        cudnn_benchmark: bool = True,
        random_seed: int = None,
        num_gpus: int = 1,
        use_wandb: bool = False):
    # General initializations
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    is_main_process = rank == MAIN_PROCESS
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Initialize logger
    extended_run_name = "{}_run-{}".format(config["name"], run_name)
    logger = Logger(
        project="glass",
        run_name=extended_run_name,
        use_wandb=use_wandb,
        config=config,
        rank=rank)

    # Setup dataset splits
    logger.info("Building datasets")
    datasets = {}

    dataset_splits = DatasetSplitter.generate_splits(config)
    transformations = TransformsGenerator.get_final_transforms(config)

    for key in dataset_splits:
        path, batching_config, split = dataset_splits[key]
        transform = transformations[key]

        datasets[key] = VideoDataset(
            path=path,
            batching_config=batching_config,
            final_transform=transform,
            allowed_videos=split,
            offset=config["data"]["offset"],
            max_num_videos=config["data"]["max_num_videos"],
            logger=logger)

    # Setup trainer
    logger.info("Instantiating trainer object")
    sampler = torch.utils.data.distributed.DistributedSampler(
        datasets["train"],
        num_replicas=num_gpus,
        rank=rank)
    trainer = Trainer(
        rank=rank,
        run_name=extended_run_name,
        config=config["training"],
        dataset=datasets["train"],
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

    # Resume training if needed
    if resume_step == -1:
        logger.info("Loading the latest checkpoint")
        trainer.load_checkpoint(model)
    elif resume_step is not None:
        logger.info(f"Loading the checkpoint at step {resume_step}")
        trainer.load_checkpoint(model, f"step_{resume_step}.pth")

    # Launch the training loop
    logger.info("Launching training loop")
    num_epochs = config["training"]["num_epochs"]
    for _ in tqdm(range(num_epochs), desc="Epochs", disable=not is_main_process):
        trainer.train_epoch(model=model, logger=logger)
