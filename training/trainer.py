import os
from tqdm import tqdm
from typing import Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import VideoDataset
from dataset.batching import single_batch_elements_collate_fn
from training.losses import (BackgroundShiftLoss,
                             BinaryMaskLoss,
                             MaskSizeLoss,
                             ForegroundTransformLoss,
                             ReconstructionLoss,
                             MultiscaleLoss,
                             PerceptualLoss)
from training.utils import calculate_motion_weights, calculate_local_motion_weights, check_ddp_consistency
from utils.configuration import Configuration
from utils.constants import MAIN_PROCESS
from utils.dict_wrapper import DictWrapper
from utils.logger import Logger
from utils.logging import (make_observations_grid,
                           to_video,
                           to_image,
                           calculate_local_action_matrix,
                           calculate_global_action_matrix)


class Trainer:
    """
    Class that handles the training
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
        Initializes the Trainer

        :param rank: rank of the current process
        :param config: training configuration
        :param dataset: dataset to train on
        :param sampler: sampler to create the dataloader with
        :param device: device to use for training
        """
        super(Trainer, self).__init__()

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

        # Training losses: GMA
        self.reconstruction_loss = \
            ReconstructionLoss().to(device)
        self.background_loss = \
            BackgroundShiftLoss().to(device)
        self.background_perceptual_loss = \
            PerceptualLoss().to(device)
        self.foreground_loss = \
            ForegroundTransformLoss().to(device)
        self.binary_mask_loss = \
            BinaryMaskLoss().to(device)
        self.mask_size_loss = \
            MaskSizeLoss(self.config["loss_params"]["mask_size"]).to(device)
        # Training losses: LMA
        self.multiscale_local_reconstruction_loss = \
            MultiscaleLoss(ReconstructionLoss()).to(device)
        self.multiscale_local_perceptual_loss = \
            MultiscaleLoss(PerceptualLoss()).to(device)
        self.local_state_reconstruction_loss = \
            nn.L1Loss().to(device)
        self.mask_reconstruction_loss = \
            nn.L1Loss().to(device)

        # Optimizer will be defined in train_epoch
        self.optimizer = None

        # Scheduler will be defined in train_epoch
        self.lr_scheduler = None

        self.global_step = 0

    def init_optimizer(self, model: nn.Module):
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config["optimizer"]["learning_rate"],
            weight_decay=self.config["optimizer"]["weight_decay"])
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            self.config["lr_schedule"],
            gamma=self.config["lr_gamma"])

    def train_epoch(
            self,
            model: nn.Module,
            logger: Logger,
            scalar_logging_frequency: int = 100,
            media_logging_frequency: int = 1000,
            saving_frequency: int = 1000):
        """
        Trains the model for one epoch
        """

        model.train()

        # Setup optimizer and scheduler if not yet
        if self.optimizer is None:
            self.init_optimizer(model)

        # Setup loading bar
        train_gen = tqdm(self.dataloader, desc="Batches", disable=not self.is_main_process, leave=False)
        for batch in train_gen:
            # Fetch data
            observations, _masks, _global_actions, _global_shifts, _local_actions, _bg_shifts, _sprite_ids = \
                batch.to_tuple()
            observations = observations.to(self.device)
            num_observations = self.calculate_num_observations()
            observations = observations[:, :num_observations]

            # Determine phase of training
            if self.global_step < self.config["pretraining_steps_gma"]:
                phase = "pretraining_gma"
            else:
                phase = "full"

            # Forward the model
            num_ground_truth_observations = self.calculate_num_gt_observations()
            gumbel_temperature = self.calculate_gumbel_temperature()
            model_outputs = model(
                observations,
                path="main",
                num_ground_truth_observations=num_ground_truth_observations,
                gumbel_temperature=gumbel_temperature)

            # Compute the loss
            loss, auxiliary_output = self.calculate_loss(model_outputs, phase)

            # Backward pass
            model.zero_grad()
            loss.backward()
            self.reduce_gradients(model, self.num_gpus)

            # Optimizer step
            self.optimizer.step()
            self.lr_scheduler.step()

            # Log scalars
            if self.global_step % scalar_logging_frequency == 0:
                for k, v in auxiliary_output.items():
                    logger.log(f"Training/Loss/{k}", v)

                # Log training stats
                logger.log("Training/Stats/num_observations", num_observations)
                logger.log("Training/Stats/num_gt_observations", num_ground_truth_observations)
                logger.log("Training/Stats/total_loss_is_nan", torch.isnan(loss).to(torch.int8))
                logger.log("Training/Stats/total_loss_is_inf", torch.isinf(loss).to(torch.int8))

            # Log media
            if self.global_step % media_logging_frequency == 0:
                # Place bounding boxes
                num_sequences = min(4, observations.size(0))
                observations_log = to_image(observations[:num_sequences])
                reconstructed_observations_log = to_image(model_outputs.reconstructed_observations[:num_sequences])

                # Log reconstructions
                logger.log("Training/Media/reconstructed_observations", logger.wandb().Image(
                    make_observations_grid(
                        observations_log,
                        reconstructed_observations_log,
                        [
                            model_outputs.foregrounds,
                            model_outputs.transformed_foregrounds,
                            model_outputs.reconstructed_foregrounds[0],
                            model_outputs.reconstructed_masks,
                            model_outputs.masks,
                            model_outputs.backgrounds,
                            model_outputs.patched_backgrounds,
                            model_outputs.inpainted_backgrounds,
                            model_outputs.shifted_backgrounds,
                        ],
                        num_sequences=num_sequences)
                ))

                # Log videos
                real_videos = to_video(observations[:num_sequences, :, :3])
                logger.log("Training/Media/real_videos", logger.wandb().Video(real_videos, fps=7))
                reconstructed_videos = to_video(model_outputs.reconstructed_observations[:num_sequences])
                logger.log("Training/Media/reconstructed_videos", logger.wandb().Video(reconstructed_videos, fps=7))
                real_foregrounds_videos = to_video(model_outputs.foregrounds[:num_sequences, :, :3])
                logger.log("Training/Media/real_foregrounds_videos",
                           logger.wandb().Video(real_foregrounds_videos, fps=7))
                reconstructed_foregrounds_videos = to_video(
                    model_outputs.reconstructed_foregrounds[0][:num_sequences])
                logger.log("Training/Media/reconstructed_foregrounds_videos",
                           logger.wandb().Video(reconstructed_foregrounds_videos, fps=7))

                # Log action matrix
                model.eval()
                action_matrix = calculate_local_action_matrix(
                    model, model_outputs.foregrounds[:num_sequences, :1], num_frames=8)
                logger.log("Training/Media/local_action_matrix", logger.wandb().Video(action_matrix, fps=6))
                model.train()

                # Log action matrix
                model.eval()
                action_matrix = calculate_global_action_matrix(
                    model, model_outputs.foregrounds[:num_sequences, :1], num_frames=8)
                logger.log("Training/Media/global_action_matrix", logger.wandb().Video(action_matrix, fps=6))
                model.train()

            # Finalize logs
            logger.finalize_logs(step=self.global_step)

            # Save checkpoint
            if self.global_step % saving_frequency == 0:
                self.save_checkpoint(model, f"step_{self.global_step}.pth")

            self.global_step += 1

        # Close loading bar
        train_gen.close()

    def calculate_loss(self, results: DictWrapper[str, Any], phase: str) -> Tuple[torch.Tensor, DictWrapper[str, Any]]:
        """
        Calculates the loss

        :param results: Dict with the model outputs
        :param phase: Phase of the model: pretraining or full
        :return: [1,] The loss value + (dict) Auxiliary output
        """

        # GMA losses

        # Reconstruction loss
        motion_weights = calculate_motion_weights(
            observations=results.observations,
            reconstructed_observations=results.reconstructed_observations,
            weight_bias=0.001)
        reconstruction_loss = self.reconstruction_loss(
            results.observations[:, 1:],
            results.reconstructed_observations,
            weights=motion_weights)

        # Transformed foregrounds l2 loss
        foreground_loss = self.foreground_loss(
            foregrounds=results.foregrounds,
            transformed_foregrounds=results.transformed_foregrounds)

        # Shifted background l1 loss
        background_loss = self.background_loss(
                backgrounds=results.inpainted_backgrounds,
                shifted_backgrounds=results.shifted_backgrounds)
        # Shifted background perceptual loss
        background_perceptual_loss = self.background_perceptual_loss(
            results.inpainted_backgrounds[:, 1:],
            results.shifted_backgrounds)

        # Loss imposing binary masks
        binary_mask_loss = self.binary_mask_loss(
            masks=results.masks)

        # Loss on the mask size
        mask_size_loss = self.mask_size_loss(
            masks=results.masks)

        # LMA losses
        foregrounds = results.foregrounds.detach()
        reconstructed_foregrounds = results.reconstructed_foregrounds

        # Calculate the motion weights
        global_motion_weights = calculate_motion_weights(foregrounds, reconstructed_foregrounds[0])
        local_motion_weights = calculate_local_motion_weights(
            foregrounds,
            results.transformed_foregrounds,
            reconstructed_foregrounds[0])

        # Reconstruction loss
        multiscale_global_reconstruction_loss = self.multiscale_local_reconstruction_loss(
            foregrounds,
            reconstructed_foregrounds,
            weights=global_motion_weights)
        multiscale_local_reconstruction_loss = self.multiscale_local_reconstruction_loss(
            foregrounds,
            reconstructed_foregrounds,
            weights=local_motion_weights)

        # Perceptual loss
        multiscale_local_perceptual_loss = self.multiscale_local_perceptual_loss(
            foregrounds,
            reconstructed_foregrounds)

        # State reconstruction loss
        local_state_reconstruction_loss = self.local_state_reconstruction_loss(
            results.local_states[:, 1:].detach(),
            results.reconstructed_local_states)

        # Action VQ loss
        local_action_vq_loss = results.local_action_vq_loss

        # Mask reconstruction loss
        mask_reconstruction_loss = self.mask_reconstruction_loss(
            results.reconstructed_masks,
            results.masks[:, 1:].detach())

        # Sum up the losses
        loss_config = self.config["loss_weights"]
        if phase == "pretraining_gma":
            multiscale_global_reconstruction_loss *= 0
            multiscale_local_reconstruction_loss *= 0
            multiscale_local_perceptual_loss *= 0
            local_state_reconstruction_loss *= 0
            local_action_vq_loss *= 0
            mask_reconstruction_loss *= 0
        # GMA losses
        loss = \
            loss_config["reconstruction_loss"] * reconstruction_loss + \
            loss_config["foreground_loss"] * foreground_loss + \
            loss_config["background_loss"] * background_loss + \
            loss_config["background_perceptual_loss"] * background_perceptual_loss + \
            loss_config["binary_mask_loss"] * binary_mask_loss + \
            loss_config["mask_size_loss"] * mask_size_loss
        # LMA losses
        loss += \
            loss_config["multiscale_global_reconstruction_loss"] * multiscale_global_reconstruction_loss + \
            loss_config["multiscale_local_reconstruction_loss"] * multiscale_local_reconstruction_loss + \
            loss_config["multiscale_local_perceptual_loss"] * multiscale_local_perceptual_loss + \
            loss_config["local_state_reconstruction_loss"] * local_state_reconstruction_loss + \
            loss_config["local_action_vq_loss"] * local_action_vq_loss + \
            loss_config["mask_reconstruction_loss"] * mask_reconstruction_loss

        # Create auxiliary output
        auxiliary_output = DictWrapper(
            # Total loss
            total_loss=loss,

            # GMA losses
            reconstruction_loss=reconstruction_loss,
            foreground_loss=foreground_loss,
            background_loss=background_loss,
            background_perceptual_loss=background_perceptual_loss,
            binary_mask_loss=binary_mask_loss,
            mask_size_loss=mask_size_loss,

            # LMA losses
            multiscale_global_reconstruction_loss=multiscale_global_reconstruction_loss,
            multiscale_local_reconstruction_loss=multiscale_local_reconstruction_loss,
            multiscale_local_perceptual_loss=multiscale_local_perceptual_loss,
            local_state_reconstruction_loss=local_state_reconstruction_loss,
            local_action_vq_loss=local_action_vq_loss,
            mask_reconstruction_loss=mask_reconstruction_loss
        )

        return loss, auxiliary_output

    def calculate_num_observations(self):
        start_num_observations = self.config["num_observations_start"]
        end_num_observations = self.config["num_observations_end"]
        start = self.config["num_observations_increase_start"]
        steps = self.config["num_observations_steps"]

        if self.global_step < start:
            return start_num_observations
        if self.global_step > start + steps:
            return end_num_observations

        scale = (self.global_step - start) / steps
        num_observations = start_num_observations + \
            int(scale * (end_num_observations - start_num_observations))

        return num_observations

    def calculate_num_gt_observations(self):
        start_num_gt_observations = self.config["gt_observations_start"]
        end_num_gt_observations = self.config["gt_observations_end"]
        start = self.config["gt_observations_decrease_start"]
        steps = self.config["gt_observations_steps"]

        if self.global_step < start:
            return start_num_gt_observations
        if self.global_step > start + steps:
            return end_num_gt_observations

        scale = (self.global_step - start) / steps
        num_gt_observations = start_num_gt_observations - \
            int(scale * (start_num_gt_observations - end_num_gt_observations))

        return num_gt_observations

    def calculate_gumbel_temperature(self):
        gumbel_temperature_start = self.config["gumbel_temperature_start"]
        gumbel_temperature_end = self.config["gumbel_temperature_end"]
        start = self.config["gumbel_temperature_decrease_start"]
        steps = self.config["gumbel_temperature_steps"]

        if self.global_step < start:
            return gumbel_temperature_start
        if self.global_step > start + steps:
            return gumbel_temperature_end

        scale = (self.global_step - start) / steps
        gumbel_temperature = gumbel_temperature_start - \
            scale * (gumbel_temperature_start - gumbel_temperature_end)

        return gumbel_temperature

    @staticmethod
    def reduce_gradients(model: nn.Module, num_gpus: int):
        params = [param for param in model.parameters() if param.grad is not None]
        if len(params) > 0:
            flat = torch.cat([param.grad.flatten() for param in params])
            if num_gpus > 1:
                torch.distributed.all_reduce(flat)
                flat /= num_gpus
            torch.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
            grads = flat.split([param.numel() for param in params])
            for param, grad in zip(params, grads):
                param.grad = grad.reshape(param.shape)

    def save_checkpoint(self, model: nn.Module, checkpoint_name: str):
        if self.num_gpus > 1:
            check_ddp_consistency(model, r".*\..+_(mean|var)")

        if self.is_main_process:
            state_dict = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "model": model.state_dict(),
                "global_step": self.global_step
            }
            torch.save(state_dict, os.path.join(self.run_path, "checkpoints", checkpoint_name))
            torch.save(state_dict, os.path.join(self.run_path, "checkpoints", "latest.pth"))

    def load_checkpoint(self, model: nn.Module, checkpoint_name: str = None):
        if checkpoint_name is None:
            checkpoint_name = "latest.pth"
        filename = os.path.join(self.run_path, "checkpoints", checkpoint_name)
        if not os.path.isfile(filename):
            raise Exception(f"Cannot load model: no checkpoint found at '{filename}'")

        # Init optimizer and scheduler if not yet
        if self.optimizer is None:
            self.init_optimizer(model)

        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        loaded_state = torch.load(filename, map_location=map_location)
        self.optimizer.load_state_dict(loaded_state["optimizer"])
        self.lr_scheduler.load_state_dict(loaded_state["lr_scheduler"])
        model.load_state_dict(loaded_state["model"])
        self.global_step = loaded_state["global_step"]
