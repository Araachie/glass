import datetime
from tqdm import tqdm
from typing import Any
import wandb

from utils.configuration import Configuration


class Logger(object):
    MAIN_PROCESS = 0

    def __init__(self, project: str, run_name: str, use_wandb: bool, config: Configuration, rank: int):
        super(Logger, self).__init__()

        if use_wandb and rank == self.MAIN_PROCESS:
            wandb.init(project=project, name=run_name, config=config)

        self.rank = rank
        self.use_wandb = use_wandb

        if use_wandb:
            self.logging_data = dict()

    def is_main_process(self):
        return self.rank == self.MAIN_PROCESS

    def info(self, data: Any):
        self._log_data_with_time("INFO", data)

    def debug(self, data: Any):
        self._log_data_with_time("DEBUG", data)

    def _log_data_with_time(self, logging_level: str, data: Any):
        if self.is_main_process():
            current_time = f"{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}"
            tqdm.write(logging_level + f" [{current_time}]: {data}")

    def log(self, field: str, data: Any):
        if self.use_wandb:
            self.logging_data[field] = data

    def finalize_logs(self, step: int):
        if self.use_wandb:
            if self.is_main_process():
                wandb.log(self.logging_data, step=step)
            self.logging_data.clear()

    @staticmethod
    def wandb():
        return wandb
