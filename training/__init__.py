from argparse import Namespace as ArgsNamespace

from .training_loop import setup_training_arguments, training_loop
from utils.distributed import setup_torch_distributed


def train(rank: int, args: ArgsNamespace, temp_dir: str):
    setup_torch_distributed(rank, args, temp_dir)

    # Execute training loop
    training_args = setup_training_arguments(args=args)
    training_loop(rank=rank, **training_args)
