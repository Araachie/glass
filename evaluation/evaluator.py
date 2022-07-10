# TODO: compress ugly code, implement metrics as separate modules

import os
import math
import numpy as np
from tqdm import tqdm

from sklearn.metrics import normalized_mutual_info_score as NMI

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import VideoDataset
from dataset.batching import single_batch_elements_collate_fn

from utils.configuration import Configuration
from utils.constants import MAIN_PROCESS
from utils.logger import Logger
from utils.tensor_folder import TensorFolder
from utils.logging import to_video


class Evaluator:
    """
    Class that handles the evaluation
    """

    def __init__(
            self,
            rank: int,
            run_name: str,
            config: Configuration,
            dataset: VideoDataset,
            sampler: torch.utils.data.distributed.Sampler,
            num_gpus: int,
            device: torch.device):
        """
        Initializes the Evaluator

        :param rank: rank of the current process
        :param config: training configuration
        :param dataset: dataset to train on
        :param sampler: sampler to create the dataloader with
        :param device: device to use for training
        """
        super(Evaluator, self).__init__()

        self.config = config
        self.rank = rank
        self.is_main_process = self.rank == MAIN_PROCESS
        self.num_gpus = num_gpus
        self.device = device

        # Create folder for saving
        self.run_path = os.path.join("runs", run_name)
        os.makedirs(self.run_path, exist_ok=True)
        os.makedirs(os.path.join(self.run_path, "checkpoints"), exist_ok=True)

        # Setup dataloader
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=dataset.batching_config["batch_size"],
            shuffle=False,
            num_workers=dataset.batching_config["num_workers"],
            sampler=sampler,
            collate_fn=single_batch_elements_collate_fn,
            pin_memory=True)

        self.global_step = 0

    @torch.no_grad()
    def evaluate(
            self,
            model: nn.Module,
            logger: Logger,
            max_num_batches: int = 1000):
        """
        Evaluates the model

        :param model: model to evaluate
        :param logger: logger used to log data
        :param max_num_batches: max number of batches to process

        """

        model.eval()

        # Setup loading bar
        validation_gen = tqdm(self.dataloader, desc="Batches", disable=not self.is_main_process, leave=False)

        # GT tensors
        gt_global_actions = None
        gt_global_shifts = None
        gt_local_actions = None
        gt_bg_shifts = None
        # Predicted tensors
        pr_bg_shifts = None
        pr_global_shifts = None
        pr_local_actions = None
        # IoU scores
        ious = None
        # NMI scores
        local_actions_per_sprite = {}
        gt_local_actions_per_sprite = {}
        # Sample reconstructions
        sample_observations = None
        sample_generated_observations = None
        # Predicted tensors RE
        pr_bg_shifts_re = None
        pr_global_shifts_re = None
        # IoU scores RE
        ious_re = None
        # IoU scores CO
        ious_co = None

        batch_size = None
        num_observations = None
        height = None
        width = None

        for i, batch in enumerate(validation_gen):
            if i >= max_num_batches:
                break

            observations, masks, global_actions, global_shifts, local_actions, bg_shifts, sprite_ids = batch.to_tuple()
            observations = observations.to(self.device)

            # Store constants
            if height is None:
                batch_size = observations.size(0)
                num_observations = observations.size(1)
                height = observations.size(3)
                width = observations.size(4)

            # Store gt actions
            global_actions = TensorFolder.flatten(global_actions[:, 1:].cpu())
            global_shifts = TensorFolder.flatten(global_shifts[:, 1:].cpu())
            flat_local_actions = TensorFolder.flatten(local_actions[:, 1:].cpu())
            bg_shifts = TensorFolder.flatten(bg_shifts[:, 1:].cpu())
            if gt_global_actions is None:
                gt_global_actions = global_actions
                gt_global_shifts = global_shifts
                gt_local_actions = flat_local_actions
                gt_bg_shifts = bg_shifts
            else:
                gt_global_actions = torch.cat([gt_global_actions, global_actions], dim=0)
                gt_global_shifts = torch.cat([gt_global_shifts, global_shifts], dim=0)
                gt_local_actions = torch.cat([gt_local_actions, flat_local_actions], dim=0)
                gt_bg_shifts = torch.cat([gt_bg_shifts, bg_shifts], dim=0)

            model_outputs = model(
                observations,
                path="main",
                num_ground_truth_observations=observations.size(1) - 1,
                gumbel_temperature=self.config["gumbel_temperature"],
                output_generated=True)

            # Save sample observations
            if sample_observations is None:
                sample_observations = observations.cpu()
                sample_generated_observations = model_outputs.generated_observations.cpu()

            # Shift scores
            m_bg_shifts = TensorFolder.flatten(model_outputs.background_shifts.cpu())
            m_global_shifts = TensorFolder.flatten(model_outputs.global_actions.cpu())
            m_local_actions = TensorFolder.flatten(
                model_outputs.local_action_ids.cpu()).squeeze(-1).squeeze(-1).squeeze(-1)
            if pr_bg_shifts is None:
                pr_bg_shifts = m_bg_shifts
                pr_global_shifts = m_global_shifts
                pr_local_actions = m_local_actions
            else:
                pr_bg_shifts = torch.cat([pr_bg_shifts, m_bg_shifts], dim=0)
                pr_global_shifts = torch.cat([pr_global_shifts, m_global_shifts], dim=0)
                pr_local_actions = torch.cat([pr_local_actions, m_local_actions], dim=0)

            # Per sprite actions
            for j in range(batch_size):
                k = f"{sprite_ids[j].cpu().detach().item()}"
                if k not in local_actions_per_sprite:
                    local_actions_per_sprite[k] = \
                        model_outputs.local_action_ids[j].cpu().squeeze(-1).squeeze(-1).squeeze(-1)
                    gt_local_actions_per_sprite[k] = local_actions[j, 1:].cpu()
                else:
                    local_actions_per_sprite[k] = torch.cat([
                        local_actions_per_sprite[k],
                        model_outputs.local_action_ids[j].cpu().squeeze(-1).squeeze(-1).squeeze(-1)
                    ], dim=0)
                    gt_local_actions_per_sprite[k] = torch.cat([
                        gt_local_actions_per_sprite[k],
                        local_actions[j, 1:].cpu()
                    ], dim=0)

            # IoU scores
            masks = masks.cpu()
            pr_masks = model_outputs.masks.cpu()
            intersections = (pr_masks * masks).sum([-1, -2])
            unions = (1 - (1 - pr_masks) * (1 - masks)).sum([-1, -2])
            m_ious = intersections / (unions + 1e-16)
            if ious is None:
                ious = m_ious
            else:
                ious = torch.cat([ious, m_ious], dim=0)

            # Reconstruciton metrics

            model_outputs_re = model(
                model_outputs.generated_observations,
                path="main",
                num_ground_truth_observations=observations.size(1) - 2,
                gumbel_temperature=self.config["gumbel_temperature"],
                output_generated=False)

            m_bg_shifts_re = TensorFolder.flatten(model_outputs_re.background_shifts.cpu())
            m_global_shifts_re = TensorFolder.flatten(model_outputs_re.global_actions.cpu())
            if pr_bg_shifts_re is None:
                pr_bg_shifts_re = m_bg_shifts_re
                pr_global_shifts_re = m_global_shifts_re
            else:
                pr_bg_shifts_re = torch.cat([pr_bg_shifts_re, m_bg_shifts_re], dim=0)
                pr_global_shifts_re = torch.cat([pr_global_shifts_re, m_global_shifts_re], dim=0)

            pr_masks_re = model_outputs_re.masks.cpu()
            intersections = (pr_masks_re * masks[:, 1:]).sum([-1, -2])
            unions = (1 - (1 - pr_masks_re) * (1 - masks[:, 1:])).sum([-1, -2])
            m_ious_re = intersections / (unions + 1e-16)
            if ious_re is None:
                ious_re = m_ious_re
            else:
                ious_re = torch.cat([ious_re, m_ious_re], dim=0)

            intersections = (pr_masks_re * pr_masks[:, 1:]).sum([-1, -2])
            unions = (1 - (1 - pr_masks_re) * (1 - pr_masks[:, 1:])).sum([-1, -2])
            m_ious_co = intersections / (unions + 1e-16)
            if ious_co is None:
                ious_co = m_ious_co
            else:
                ious_co = torch.cat([ious_co, m_ious_co], dim=0)

        global_nmi = NMI(gt_local_actions.detach().numpy(), pr_local_actions.detach().numpy())
        logger.info(f"Global NMI: {global_nmi}")
        local_nmi = np.mean([
            NMI(gt_local_actions_per_sprite[k].detach().numpy(), local_actions_per_sprite[k].detach().numpy())
            for k in local_actions_per_sprite.keys()])
        logger.info(f"Local NMI: {local_nmi}")
        random_nmi = NMI(
            np.random.randint(0, 6, size=pr_local_actions.detach().numpy().shape[0]),
            gt_local_actions.detach().numpy())
        logger.info(f"Random NMI: {random_nmi}")

        # Background shift metric
        pr_bg_shifts[:, 0] = -pr_bg_shifts[:, 0] * width * 0.5
        pr_bg_shifts[:, 1] = -pr_bg_shifts[:, 1] * height * 0.5
        bg_shift_diff = gt_bg_shifts - pr_bg_shifts
        bg_shift_rmse = torch.norm(bg_shift_diff, p=2, dim=1)
        non_zero_indices = torch.norm(gt_bg_shifts, p=2, dim=1) > 0
        pr_bg_shifts_non_zero = pr_bg_shifts[non_zero_indices]
        gt_bg_shifts_non_zero = gt_bg_shifts[non_zero_indices]
        bg_shift_cos = (gt_bg_shifts_non_zero * pr_bg_shifts_non_zero).sum(1) / \
            torch.norm(gt_bg_shifts_non_zero, p=2, dim=1) / \
            torch.norm(pr_bg_shifts_non_zero, p=2, dim=1)
        bg_shift_angle = (bg_shift_cos > (1 / math.sqrt(2))).to(torch.float32)

        # Global shift metric
        pr_global_shifts[:, 0] = pr_global_shifts[:, 0] * width * 0.5
        pr_global_shifts[:, 1] = pr_global_shifts[:, 1] * height * 0.5
        global_shifts_diff = gt_global_shifts - pr_global_shifts
        global_shift_rmse = torch.norm(global_shifts_diff, p=2, dim=1)
        non_zero_indices = torch.norm(gt_global_shifts, p=2, dim=1) > 0
        gt_global_shifts_non_zero = gt_global_shifts[non_zero_indices]
        pr_global_shifts_non_zero = pr_global_shifts[non_zero_indices]
        global_shift_cos = (gt_global_shifts_non_zero * pr_global_shifts_non_zero).sum(1) / \
            torch.norm(gt_global_shifts_non_zero, p=2, dim=1) / \
            torch.norm(pr_global_shifts_non_zero, p=2, dim=1)
        global_shift_angle = (global_shift_cos > (1 / math.sqrt(2))).to(torch.float32)

        logger.info(f"BG shift mean norm: {bg_shift_rmse.mean()}")
        logger.info(f"BG shift min norm: {bg_shift_rmse.min()}")
        logger.info(f"BG shift max norm: {bg_shift_rmse.max()}")
        logger.info(f"BG angle match: {bg_shift_angle.mean()}")

        logger.info(f"Global shift mean norm: {global_shift_rmse.mean()}")
        logger.info(f"Global shift min norm: {global_shift_rmse.min()}")
        logger.info(f"Global shift max norm: {global_shift_rmse.max()}")
        logger.info(f"Global angle match: {global_shift_angle.mean()}")

        logger.info(f"Mean IoU: {ious.mean()}")

        # Background shift metric
        gt_bg_shifts = gt_bg_shifts.view(max_num_batches, batch_size, num_observations - 1, 2)
        gt_bg_shifts = gt_bg_shifts[:, :, 1:].reshape(max_num_batches * batch_size * (num_observations - 2), 2)
        pr_bg_shifts_re[:, 0] = -pr_bg_shifts_re[:, 0] * width * 0.5
        pr_bg_shifts_re[:, 1] = -pr_bg_shifts_re[:, 1] * height * 0.5
        bg_shift_diff_re = gt_bg_shifts - pr_bg_shifts_re
        bg_shift_rmse_re = torch.norm(bg_shift_diff_re, p=2, dim=1)
        non_zero_indices = torch.norm(gt_bg_shifts, p=2, dim=1) > 0
        gt_bg_shifts_non_zero = gt_bg_shifts[non_zero_indices]
        pr_bg_shifts_re_non_zero = pr_bg_shifts_re[non_zero_indices]
        bg_shift_cos_re = (gt_bg_shifts_non_zero * pr_bg_shifts_re_non_zero).sum(1) / \
            torch.norm(gt_bg_shifts_non_zero, p=2, dim=1) / \
            torch.norm(pr_bg_shifts_re_non_zero, p=2, dim=1)
        bg_shift_angle_re = (bg_shift_cos_re > (1 / math.sqrt(2))).to(torch.float32)

        # Global shift metric
        gt_global_shifts = gt_global_shifts.view(max_num_batches, batch_size, num_observations - 1, 2)
        gt_global_shifts = gt_global_shifts[:, :, 1:].reshape(max_num_batches * batch_size * (num_observations - 2), 2)
        pr_global_shifts_re[:, 0] = pr_global_shifts_re[:, 0] * width * 0.5
        pr_global_shifts_re[:, 1] = pr_global_shifts_re[:, 1] * height * 0.5
        global_shifts_diff_re = gt_global_shifts - pr_global_shifts_re
        global_shift_rmse_re = torch.norm(global_shifts_diff_re, p=2, dim=1)
        non_zero_indices = torch.norm(gt_global_shifts, p=2, dim=1) > 0
        gt_global_shifts_non_zero = gt_global_shifts[non_zero_indices]
        pr_global_shifts_re_non_zero = pr_global_shifts_re[non_zero_indices]
        global_shift_cos_re = (gt_global_shifts_non_zero * pr_global_shifts_re_non_zero).sum(1) / \
            torch.norm(gt_global_shifts_non_zero, p=2, dim=1) / \
            torch.norm(pr_global_shifts_re_non_zero, p=2, dim=1)
        global_shift_angle_re = (global_shift_cos_re > (1 / math.sqrt(2))).to(torch.float32)

        logger.info(f"BG shift mean norm RE: {bg_shift_rmse_re.mean()}")
        logger.info(f"BG shift min norm RE: {bg_shift_rmse_re.min()}")
        logger.info(f"BG shift max norm RE: {bg_shift_rmse_re.max()}")
        logger.info(f"BG angle match RE: {bg_shift_angle_re.mean()}")

        logger.info(f"Global shift mean norm RE: {global_shift_rmse_re.mean()}")
        logger.info(f"Global shift min norm RE: {global_shift_rmse_re.min()}")
        logger.info(f"Global shift max norm RE: {global_shift_rmse_re.max()}")
        logger.info(f"Global angle match RE: {global_shift_angle_re.mean()}")

        logger.info(f"Mean IoU RE: {ious_re.mean()}")

        # Background shift metric
        pr_bg_shifts = pr_bg_shifts.view(max_num_batches, batch_size, num_observations - 1, 2)
        pr_bg_shifts = pr_bg_shifts[:, :, 1:].reshape(max_num_batches * batch_size * (num_observations - 2), 2)
        bg_shift_diff_co = pr_bg_shifts - pr_bg_shifts_re
        bg_shift_rmse_co = torch.norm(bg_shift_diff_co, p=2, dim=1)
        non_zero_indices = torch.norm(pr_bg_shifts, p=2, dim=1) > 0
        pr_bg_shifts_non_zero = pr_bg_shifts[non_zero_indices]
        pr_bg_shifts_re_non_zero = pr_bg_shifts_re[non_zero_indices]
        bg_shift_cos_co = (pr_bg_shifts_non_zero * pr_bg_shifts_re_non_zero).sum(1) / torch.norm(
            pr_bg_shifts_non_zero, p=2, dim=1) / torch.norm(
            pr_bg_shifts_re_non_zero, p=2, dim=1)
        bg_shift_angle_co = (bg_shift_cos_co > (1 / math.sqrt(2))).to(torch.float32)

        # Global shift metric
        pr_global_shifts = pr_global_shifts.view(max_num_batches, batch_size, num_observations - 1, 2)
        pr_global_shifts = pr_global_shifts[:, :, 1:].reshape(max_num_batches * batch_size * (num_observations - 2), 2)
        global_shifts_diff_co = pr_global_shifts - pr_global_shifts_re
        global_shift_rmse_co = torch.norm(global_shifts_diff_co, p=2, dim=1)
        non_zero_indices = torch.norm(pr_global_shifts, p=2, dim=1) > 0
        pr_global_shifts_non_zero = pr_global_shifts[non_zero_indices]
        pr_global_shifts_re_non_zero = pr_global_shifts_re[non_zero_indices]
        global_shift_cos_co = (pr_global_shifts_non_zero * pr_global_shifts_re_non_zero).sum(1) / torch.norm(
            pr_global_shifts_non_zero, p=2, dim=1) / torch.norm(
            pr_global_shifts_re_non_zero, p=2, dim=1)
        global_shift_angle_co = (global_shift_cos_co > (1 / math.sqrt(2))).to(torch.float32)

        logger.info(f"BG shift mean norm CO: {bg_shift_rmse_co.mean()}")
        logger.info(f"BG shift min norm CO: {bg_shift_rmse_co.min()}")
        logger.info(f"BG shift max norm CO: {bg_shift_rmse_co.max()}")
        logger.info(f"BG angle match CO: {bg_shift_angle_co.mean()}")

        logger.info(f"Global shift mean norm CO: {global_shift_rmse_co.mean()}")
        logger.info(f"Global shift min norm CO: {global_shift_rmse_co.min()}")
        logger.info(f"Global shift max norm CO: {global_shift_rmse_co.max()}")
        logger.info(f"Global angle match CO: {global_shift_angle_co.mean()}")

        logger.info(f"Mean IoU CO: {ious_co.mean()}")

        logger.log(f"real_videos", logger.wandb().Video(to_video(sample_observations[:4, :, :3]), fps=7))
        logger.log(f"generated_videos", logger.wandb().Video(to_video(sample_generated_observations[:4, :, :3]), fps=7))

        logger.finalize_logs(step=0)

    def load_checkpoint(self, model: nn.Module, checkpoint_name: str = None):
        distributed = isinstance(model, torch.nn.parallel.DistributedDataParallel)
        if checkpoint_name is None:
            checkpoint_name = "latest.pth"
        filename = os.path.join(self.run_path, "checkpoints", checkpoint_name)
        if not os.path.isfile(filename):
            raise Exception(f"Cannot load model: no checkpoint found at '{filename}'")

        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        loaded_state = torch.load(filename, map_location=map_location)
        model_state = loaded_state["model"]
        if not distributed and list(model_state.keys())[0].startswith("module"):
            model_state = {".".join(k.split(".")[1:]): v for k, v in model_state.items()}
        elif distributed and not list(model_state.keys())[0].startswith("module"):
            model_state = {f"module.{k}": v for k, v in model_state.items()}
        model.load_state_dict(model_state)
        self.global_step = loaded_state["global_step"]
