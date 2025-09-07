from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from tnp.data.base import MultiModalBatch
from tnp.data.era5.era5 import (
    BaseERA5Batch,
    BaseERA5DataGeneratorWithReset,
    Batch,
    ERA5DataGenerator,
)
from tnp.data.era5.normalisation import locations as var_means
from tnp.data.era5.normalisation import scales as var_stds


@dataclass
class ERA5MultiModalBatch(MultiModalBatch, BaseERA5Batch, Batch):
    time_grid: Optional[torch.Tensor] = None


class MultiModalERA5DataGenerator(ERA5DataGenerator):

    def sample_latent_time_grid(self, x_grid: torch.Tensor, latent_time_steps: Optional[int] = None) -> torch.Tensor:
        """
        Generate a time grid that interpolates latent time steps within the T dimension.
        
        Args:
            x_grid: Input grid of shape [B, T, H, W, N] where N contains [time, lat, lon]
            latent_time_steps: Number of latent time steps to generate. If None, uses self.latent_time_steps
            
        Returns:
            time_grid: Tensor of shape [B, L_T] with interpolated time values
        """

        if latent_time_steps is None and self.latent_time_steps is not None:
            latent_time_steps = self.latent_time_steps

        else: #was not set in the constructor
            return x_grid[..., 0, 0, 0]
            
        # Extract time values from x_grid - shape [B, T]
        timestamps = x_grid[..., 0, 0, 0]  # [B, T]
        _, original_time_steps = timestamps.shape
        
        assert latent_time_steps < original_time_steps, "latent_time_steps must be less than or equal to original_time_steps"
        
        # Use F.interpolate for compact linear interpolation
        # Add channel dimension, interpolate, then squeeze
        return F.interpolate(
            timestamps.unsqueeze(1),  # [B, 1, T]
            size=latent_time_steps,
            mode='linear',
            align_corners=False  # This ensures even spacing without forcing endpoints
        ).squeeze(1) # [B, latent_time_steps]

    def sample_batch(
        self,
        pc: float,
        pt: float,
        x_grid: torch.Tensor,
        data_vars: Dict[str, torch.Tensor],
        target_time_slice: Optional[slice] = None,
        latent_time_steps: Optional[int] = None,
    ) -> ERA5MultiModalBatch:
        

        if not self.use_time:
            time_grid = None
            data_vars = {k: data_vars[k].squeeze(1) for k in self.data_vars}
        else:
            time_grid = self.sample_latent_time_grid(x_grid, latent_time_steps)

        print(f"MultiModalERA5DataGenerator: (latent) time_grid[0]: {time_grid[0]} of shape {time_grid.shape}")
        print(f"MultiModalERA5DataGenerator: x_grid shape: {x_grid.shape}")

        # Apply target_time_slice to filter out specific time steps from targets
        if target_time_slice is not None:
            assert target_time_slice[0] >= 0 and target_time_slice[1] <= time_grid.shape[1], "target_time_slice must be within the range of the time_grid"
            # output_time_grid = time_grid[..., target_time_slice]
            # Create filtered grids for target sampling
            x_grid_target = x_grid[..., target_time_slice, :, :, :]
            data_vars_target = {k: v[..., target_time_slice, :, :] for k, v in data_vars.items()}
            # B, T, H, W, N = x_grid_target.shape
        else:
            x_grid_target = x_grid
            data_vars_target = data_vars
            # B, T, H, W, N = x_grid.shape

        xc: Dict[str, torch.Tensor] = {}
        yc: Dict[str, torch.Tensor] = {}
        for var, y_grid in data_vars.items():

            # Sample context mask.
            mc_grid_idx = self.sample_masks(
                prop=pc, grid_shape=y_grid.shape[1:], batch_shape=y_grid.shape[0]
            )

            batch_idx = torch.arange(x_grid.shape[0]).unsqueeze(-1)
            xc[var] = x_grid[(batch_idx,) + mc_grid_idx]
            yc[var] = y_grid[(batch_idx,) + mc_grid_idx][..., None]

        # Stack data_vars_target into a single tensor for target sampling.
        y_grid_target = torch.stack([data_vars_target[k] for k in self.data_vars], dim=-1)

        # Sample target mask from the filtered target grid.
        mt_grid_idx = self.sample_masks(
            prop=pt, grid_shape=y_grid_target.shape[1:-1], batch_shape=y_grid_target.shape[0]
        )
        batch_idx = torch.arange(x_grid_target.shape[0]).unsqueeze(-1)
        xt = x_grid_target[(batch_idx,) + mt_grid_idx]
        yt = y_grid_target[(batch_idx,) + mt_grid_idx]
        print(f"MultiModalERA5DataGenerator: xt shape: {xt.shape}, yt shape: {yt.shape}")
        
        # Flatten x and y using original full grids.
        x = x_grid.flatten(start_dim=1, end_dim=-2)
        y_grid = torch.stack([data_vars[k] for k in self.data_vars], dim=-1)
        y = y_grid.flatten(start_dim=1, end_dim=-2)

        return ERA5MultiModalBatch(
            x=x,
            y=y,
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
            lat_range=self.lat_range,
            lon_range=self.lon_range,
            var_names=self.data_vars,
            var_means=var_means,
            var_stds=var_stds,
            time_grid=time_grid, #latent grid
        )


class MultiModalERA5DataGeneratorWithReset(
    MultiModalERA5DataGenerator, BaseERA5DataGeneratorWithReset
):
    pass
