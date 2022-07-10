# Most of the evaluation scripts were adapted
# from this repository https://github.com/willi-menapace/PlayableVideoGeneration

from argparse import Namespace as ArgsNamespace

from .evaluation import setup_evaluation_arguments, launch_evaluation
from utils.distributed import setup_torch_distributed


def evaluate(rank: int, args: ArgsNamespace, temp_dir: str):
    setup_torch_distributed(rank, args, temp_dir)

    # Execute training loop
    evaluation_args = setup_evaluation_arguments(args=args)
    launch_evaluation(rank=rank, **evaluation_args)
