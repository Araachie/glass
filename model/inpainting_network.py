from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import SameBlock, UpBlock

from utils.configuration import Configuration
from utils.dict_wrapper import DictWrapper
from utils.tensor_folder import TensorFolder


class InpaintingNetwork(nn.Module):
    """
    Model that inpaints backgrounds
    """

    def __init__(self, config: Configuration):
        super(InpaintingNetwork, self).__init__()

        self.config = config

        self.inpainting_network_down = nn.Sequential(
            SameBlock(in_planes=3, out_planes=16, downsample_factor=2),  # res / 2
            SameBlock(in_planes=16, out_planes=32, downsample_factor=2),  # res / 4
            SameBlock(in_planes=32, out_planes=32),  # res / 4
            SameBlock(in_planes=32, out_planes=64, downsample_factor=2),  # res / 8
            SameBlock(in_planes=64, out_planes=64),  # res / 8
        )
        self.inpainting_network_up = nn.Sequential(
            SameBlock(in_planes=64, out_planes=64),  # res / 8
            UpBlock(in_features=64, out_features=32, scale_factor=2, upscaling_mode="bilinear"),  # res / 4
            UpBlock(in_features=32, out_features=32, scale_factor=2, upscaling_mode="bilinear"),  # res / 2
            UpBlock(in_features=32, out_features=16, scale_factor=2, upscaling_mode="bilinear"),  # res
            SameBlock(in_planes=16, out_planes=3),  # res
        )

        self.out_of_domain_value = -1.1

    @staticmethod
    def draw_circles(images: torch.Tensor, centers: torch.Tensor, radiuses: torch.Tensor) -> torch.Tensor:
        """

        :param images: (bs, h, w)
        :param centers: (bs, 2)
        :param radiuses: (bs, 1)
        :return:
        """
        h, w = images.size(1), images.size(2)

        x_range = torch.arange(w)
        y_range = torch.arange(h)

        yy, xx = torch.meshgrid([y_range, x_range], indexing="ij")
        coords = torch.stack([xx, yy], dim=-1).to(images.device)

        rs = torch.pow(coords[:, :, 0] - centers[:, 0].unsqueeze(-1).unsqueeze(-1), 2) + \
            torch.pow(coords[:, :, 1] - centers[:, 1].unsqueeze(-1).unsqueeze(-1), 2)
        Rs = torch.pow(radiuses, 2).unsqueeze(-1)

        images[(rs < Rs)] = 1.0

        return images

    @staticmethod
    def generate_random_mask(thetas: torch.Tensor, height2: int = 96, width2: int = 128) -> torch.Tensor:
        """

        :param thetas: (bs, n_strokes, 8)
        :param height2: int height
        :param width2:
        :return:
        """
        n_masks = thetas.size(0)

        # thetas[:, 0:2]: starting point (x0, y0)
        # thetas[:, 2:4]: ending point (x2, y2)
        thetas[:, :, 0] *= width2
        thetas[:, :, 2] *= width2
        thetas[:, :, 1] *= height2
        thetas[:, :, 3] *= height2

        # thetas[:, 4:6]: middle point, it stay between starting points and ending points
        thetas[:, :, 4:6] = thetas[:, :, 0:2] + (thetas[:, :, 2:4] - thetas[:, :, 0:2]) * thetas[:, :, 4:6]

        # thetas[:, 6:8]: the thickness of the strokes(mask)
        # add 2 to ensure the strokes are not to thin.
        thetas[:, :, 6:8] = thetas[:, :, 6:8] * (width2 // 8) + 2

        # draw the random strokes(mask)
        canvas = torch.zeros([n_masks, height2, width2], dtype=torch.uint8).to(thetas.device)
        thetas = thetas.permute(1, 2, 0)
        gap = width2 // 2
        for idx, para in enumerate(thetas):
            x0, y0, x2, y2, x1, y1, z0, z2 = para
            for p in range(gap):  # bezier curve
                p /= gap
                q = 1 - p

                pp = p * p
                qq = q * q
                pq2 = p * q * 2

                x = (pp * x2 + pq2 * x1 + qq * x0).to(torch.long)
                y = (pp * y2 + pq2 * y1 + qq * y0).to(torch.long)
                z = (p * z2 + q * z0).to(torch.long)

                centers = torch.stack([x, y], dim=-1)
                radiuses = z.view(-1, 1)
                canvas = InpaintingNetwork.draw_circles(canvas, centers, radiuses)  # img, center, radius

        return canvas

    def forward(
            self,
            observations: torch.Tensor,
            masks: torch.Tensor,
            randomize_masks: bool = False,
            combine: bool = True,
            enlarge_masks: bool = False) -> DictWrapper[str, Any]:
        """
        Inpaints backgrounds

        :param observations: (bs, num_observations, 3, height, width)
        :param masks: (bs, num_observations, 1, height, width)
        :param randomize_masks: bool, whether to unite masks with some random masks
        :param combine: bool
        :param enlarge_masks: bool, whether to add some pixels to the masks or not
        :return:
        """

        batch_size = observations.size(0)
        num_observations = observations.size(1)
        height = observations.size(3)
        width = observations.size(4)

        # Generate random masks
        if randomize_masks:
            random_masks = self.generate_random_mask(
                torch.rand(batch_size * num_observations, 1, 8).to(masks.device),
                height2=height,
                width2=width)
            random_masks = TensorFolder.fold(random_masks, num_observations).unsqueeze(2)
        else:
            random_masks = torch.zeros(
                [batch_size, num_observations, 1, height, width]).to(torch.float32).to(masks.device)
        union_masks = 1 - (1 - masks) * (1 - random_masks)
        flat_union_masks = TensorFolder.flatten(union_masks)

        # Add some pixels if needed
        if enlarge_masks:
            flat_union_masks = F.max_pool2d(flat_union_masks, kernel_size=3, padding=1)

        # Patch backgrounds with out of domain value
        patched_backgrounds = self.out_of_domain_value * union_masks + observations * (1 - union_masks)
        flat_patched_backgrounds = TensorFolder.flatten(patched_backgrounds)

        # Inpaint backgrounds
        flat_inpainted_backgrounds = self.inpainting_network_up(
            self.inpainting_network_down(flat_patched_backgrounds))
        folded_inpainted_backgrounds = TensorFolder.fold(flat_inpainted_backgrounds, num_observations)

        # Combine original with inpainted
        if combine:
            folded_inpainted_backgrounds = \
                observations * (1 - union_masks) + folded_inpainted_backgrounds * union_masks

        return DictWrapper(
            patched_backgrounds=patched_backgrounds,
            inpainted_backgrounds=folded_inpainted_backgrounds)
