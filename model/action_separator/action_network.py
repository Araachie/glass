import torch
import torch.nn as nn
import torch.nn.functional as F

from model.action_separator.quantizer import VectorQuantizer
from model.layers.residual_block import ResidualBlock
from utils.configuration import Configuration
from utils.dict_wrapper import DictWrapper
from utils.tensor_folder import TensorFolder


class ActionNetwork(nn.Module):
    """
    Model that predicts the actions given the sequence of observations
    """

    def __init__(
            self,
            state_size: int,
            num_actions: int,
            action_space_dimension: int,
            use_gumbel: bool = False,
            use_plain_direction: bool = False):
        super(ActionNetwork, self).__init__()

        self.state_size = state_size
        self.action_space_dimension = action_space_dimension
        self.num_actions = num_actions
        self.use_plain_direction = use_plain_direction
        self.use_gumbel = use_gumbel

        residual_blocks = [
            ResidualBlock(self.state_size, 2 * self.state_size, downsample_factor=2),
            ResidualBlock(2 * self.state_size, self.state_size, downsample_factor=1),
        ]
        self.residuals = nn.Sequential(*residual_blocks)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Mapping network
        layers = []
        num_layers = 2
        for i in range(num_layers):
            in_size = self.state_size if i == 0 else 2 * self.state_size
            out_size = self.state_size if i == num_layers - 1 else 2 * self.state_size
            layers.append(nn.Linear(in_size, out_size))

            if i != num_layers - 1:
                layers.append(nn.BatchNorm1d(out_size))
                layers.append(nn.LeakyReLU(0.2))
        self.mapping = nn.Sequential(*layers)

        # Projection layer to map second feature vector to the first
        self.projections = nn.ModuleList([
            nn.Bilinear(self.state_size, self.state_size, 2 * self.state_size),
            nn.Bilinear(self.state_size, 2 * self.state_size, 2 * self.state_size),
            nn.Bilinear(self.state_size, 2 * self.state_size, 2 * self.state_size),
            nn.Bilinear(self.state_size, 2 * self.state_size, self.state_size),
        ])
        self.affines = nn.ModuleList([
            nn.Linear(self.state_size, self.state_size),
            nn.Linear(self.state_size, self.state_size),
            nn.Linear(self.state_size, self.state_size),
            nn.Linear(self.state_size, self.state_size),
        ])
        self.bns_acts = nn.ModuleList([
            nn.Sequential(nn.BatchNorm1d(2 * self.state_size), nn.LeakyReLU(0.2)),
            nn.Sequential(nn.BatchNorm1d(2 * self.state_size), nn.LeakyReLU(0.2)),
            nn.Sequential(nn.BatchNorm1d(2 * self.state_size), nn.LeakyReLU(0.2)),
            None
        ])

        # Linear layers for the prediction of the action codes
        if use_gumbel:
            self.final_mlp = nn.Linear(self.state_size, self.num_actions)
        else:
            self.final_mlp = nn.Sequential(
                nn.Linear(self.state_size, 2 * self.state_size),
                nn.BatchNorm1d(2 * self.state_size),
                nn.LeakyReLU(0.2),
                nn.Linear(2 * self.state_size, self.state_size),
                nn.BatchNorm1d(self.state_size),
                nn.LeakyReLU(0.2),
                nn.Linear(self.state_size, self.action_space_dimension))

        self.action_codes_quantizer = VectorQuantizer(
            n_e=self.num_actions,
            e_dim=self.action_space_dimension,
            beta=0.25)

    def forward(
            self,
            states: torch.Tensor,
            states_attention: torch.Tensor,
            next_states: torch.Tensor,
            next_states_attention: torch.Tensor,
            gumbel_temperature: float) -> DictWrapper[str, torch.Tensor]:
        """
        Computes actions corresponding to the state transition from predecessor to successor state

        :param states: (bs, num_observations, states_features, states_height, states_width) tensor with states
        :param states_attention: (bs, num_observations, 1, states_height, states_width) tensor with attention
        :param next_states: (bs, num_observations, states_features, states_height, states_width) tensor with states
        :param next_states_attention: (bs, num_observations, 1, states_height, states_width) tensor with attention
        :param gumbel_temperature: TAU in gumbel softmax trick
        :return:

        """

        num_observations = states.size(1)

        # Applies attention
        attentive_states = states * states_attention
        flat_attentive_states = TensorFolder.flatten(attentive_states)
        next_attentive_states = next_states * next_states_attention
        next_flat_attentive_states = TensorFolder.flatten(next_attentive_states)

        # Convolutional blocks
        x = self.residuals(flat_attentive_states)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        next_x = self.residuals(next_flat_attentive_states)
        next_x = self.gap(next_x).squeeze(-1).squeeze(-1)

        # Calculate directions
        folded_x = TensorFolder.fold(x, num_observations)  # [bs, no, s]
        next_folded_x = TensorFolder.fold(next_x, num_observations)  # [bs, no, s]
        folded_directions = self.calculate_directions(folded_x, next_folded_x)  # [bs, no, s]
        flat_directions = TensorFolder.flatten(folded_directions)  # [bs * no, s]

        # Final mlp
        if self.use_gumbel:
            flat_action_logits = self.final_mlp(flat_directions)  # [bs * no, na]
            flat_action_logprobs = F.log_softmax(flat_action_logits, dim=1)
            flat_actions = F.gumbel_softmax(flat_action_logprobs, tau=gumbel_temperature, hard=False)
            flat_action_codes = torch.matmul(flat_actions, self.action_codes_quantizer.embedding.weight)
            folded_action_codes = \
                TensorFolder.fold(flat_action_codes, num_observations)  # [bs, no, as]
            folded_actions = TensorFolder.fold(flat_actions, num_observations)
            return DictWrapper(
                vq_loss=0,
                action_codes=folded_action_codes,
                action_ids=folded_actions)
        else:
            flat_action_codes = self.final_mlp(flat_directions)  # [bs * no, as]
            vq_loss, flat_quantized_action_codes, _, _, flat_action_ids, _ = \
                self.action_codes_quantizer(flat_action_codes.unsqueeze(2).unsqueeze(3))
            flat_quantized_action_codes = flat_quantized_action_codes.squeeze(3).squeeze(2)  # [bs * no, as]
            folded_quantized_action_codes = \
                TensorFolder.fold(flat_quantized_action_codes, num_observations)  # [bs, no, as]
            folded_action_ids = TensorFolder.fold(flat_action_ids, num_observations)

            return DictWrapper(
                vq_loss=vq_loss,
                action_codes=folded_quantized_action_codes,
                action_ids=folded_action_ids)

    def calculate_directions(self, first: torch.Tensor, second: torch.Tensor):
        """

        :param first: (bs, num_observations, state_size)
        :param second: (bs, num_observations, state_size)
        :return: (bs, num_observations, state_size)
        """

        if self.use_plain_direction:
            return second - first

        num_observations = first.size(1)

        flat_first = TensorFolder.flatten(first)
        flat_second = TensorFolder.flatten(second)

        projected_second = flat_second
        for projection_layer, affine, bn_act in zip(self.projections, self.affines, self.bns_acts):
            projected_second = projection_layer(affine(flat_first), projected_second)
            if bn_act is not None:
                projected_second = bn_act(projected_second)

        directions = projected_second - flat_first

        return TensorFolder.fold(directions, num_observations)


def build_action_network(config: Configuration) -> nn.Module:
    return ActionNetwork(
        state_size=config["state_size"],
        num_actions=config["num_actions"],
        action_space_dimension=config["action_space_dimension"],
        use_gumbel=config["use_gumbel"],
        use_plain_direction=config["use_plain_direction"])
