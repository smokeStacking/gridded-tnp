import math
from abc import ABC
from typing import Tuple

import numpy as np
import torch
from torch import nn

from ..utils.spherical_harmonics.spherical_harmonics_ylm import spherical_harmonics


class Embedding(nn.Module, ABC):
    def __init__(self, active_dims: Tuple[int, ...]):
        super().__init__()

        # Which dimensions to apply the embedding to.
        self.active_dims = active_dims


class SineActivation(nn.Module):
    def __init__(self, w0: float = 1.0):
        super().__init__()

        # TODO: need to change for first layer?
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class SphericalHarmonicsEmbedding(Embedding):
    def __init__(
        self,
        active_dims: Tuple[int, ...],
        num_legendre_polys: int = 10,
        lonlat_dims: Tuple[int, ...] = (-1, -2),
    ):
        super().__init__(active_dims)

        self.num_legendre_polys = int(num_legendre_polys)
        self.embed_dim = self.num_legendre_polys**2
        self.lonlat_dims = lonlat_dims
        self.spherical_harmonics = spherical_harmonics

    def forward(self, x: torch.Tensor):
        """
        Assumes x[..., lonlat_dims] = (lon, lat) where lon is in [-180, 180] and lat is in
        [-90, 90].
        """

        if x.shape[-1] > 2:
            x_other = torch.stack(
                [
                    x[..., dim]
                    for dim in range(x.shape[-1])
                    if dim not in self.lonlat_dims
                ]
            )
        else:
            x_other = None

        lon, lat = x[..., self.lonlat_dims[0]], x[..., self.lonlat_dims[1]]

        # Assumes phi is in [-pi, pi] and lat is in [-pi / 2, pi / 2].
        phi, theta = torch.deg2rad(lon), torch.deg2rad(lat)

        # Compute the spherical harmonics.
        sh_list = []
        for l in range(self.num_legendre_polys):
            for m in range(-l, l + 1):
                sh = self.spherical_harmonics(m, l, phi, theta)
                if isinstance(sh, float):
                    sh = sh * torch.ones_like(phi)
                sh_list.append(sh)

        out = torch.stack(sh_list, dim=-1)

        if x_other is not None:
            out = torch.cat((x_other, out), dim=-1)

        return out


class FourierEmbedding(Embedding):
    def __init__(
        self,
        lower: float,
        upper: float,
        active_dim: int,
        assert_range: bool = True,
        num_wavelengths: int = 10,
    ):
        super().__init__((active_dim,))

        self.lower = lower
        self.upper = upper
        self.assert_range = assert_range
        self.num_wavelengths = num_wavelengths

        # We will use half of the dimensionality for `sin` and the other half for `cos`.
        if num_wavelengths % 2 != 0:
            raise ValueError("The dimensionality must be a multiple of two.")

    def forward(self, x: torch.Tensor):
        # If the input is not within the configured range, the embedding might be ambiguous!
        in_range = torch.logical_and(
            self.lower <= x.abs(), torch.all(x.abs() <= self.upper)
        )
        in_range_or_zero = torch.all(
            torch.logical_or(in_range, x == 0)
        )  # Allow zeros to pass through.
        if self.assert_range and not in_range_or_zero:
            raise AssertionError(
                f"The input tensor is not within the configured range"
                f" `[{self.lower}, {self.upper}]`."
            )

        # Always perform the expansion with `float64`s to avoid numerical accuracy shenanigans.
        x = x.double()

        wavelengths = torch.logspace(
            math.log10(self.lower),
            math.log10(self.upper),
            self.num_wavelengths // 2,
            base=10,
            device=x.device,
            dtype=x.dtype,
        )
        prod = x * 2 * np.pi / wavelengths
        encoding = torch.cat((torch.sin(prod), torch.cos(prod)), dim=-1)

        return encoding.float()  # Cast to `float32` to avoid incompatibilities.


class PressureFourierEncoder(nn.Module):
    """
    Encoder for pressure levels using Fourier features.
    
    Applies Fourier embedding to pressure values to capture the hierarchical
    structure of the atmosphere across pressure levels.
    
    Args:
        num_wavelengths: Number of Fourier wavelengths (must be even)
        pressure_range: Tuple of (min, max) pressure in hPa
    """
    def __init__(self, num_wavelengths: int = 10, pressure_range: tuple = (1.0, 1013.0)):
        super().__init__()
        self.num_wavelengths = num_wavelengths
        self.pressure_range = pressure_range
        
        self.fourier = FourierEmbedding(
            lower=pressure_range[0], 
            upper=pressure_range[1],
            active_dim=0,  # pressure is 1D scalar
            num_wavelengths=num_wavelengths,
            assert_range=False  # Don't assert range to allow flexibility
        )
    
    def forward(self, pressure: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier embedding to pressure values.
        
        Args:
            pressure: Pressure tensor of shape [B, N, L] or [B, N, 1] or [B, N]
                     where L is number of levels
        
        Returns:
            Fourier embedded pressure:
            - For 3D: [B, N, L, num_wavelengths] (per-level embeddings)
            - For 2D: [B, N, num_wavelengths]
        """
        if pressure.dim() == 3 and pressure.shape[-1] > 1:
            # Shape [B, N, L] - apply Fourier to each level independently
            # Output: [B, N, L, num_wavelengths] for per-level processing
            B, N, L = pressure.shape
            pressure = pressure.reshape(B * N * L, 1)  # Flatten all for Fourier
            p_embed = self.fourier(pressure)  # [B*N*L, num_wavelengths]
            p_embed = p_embed.reshape(B, N, L, self.num_wavelengths)  # Keep level dimension
            return p_embed
        elif pressure.dim() == 3 and pressure.shape[-1] == 1:
            # Shape [B, N, 1] - apply Fourier
            pressure = pressure.squeeze(-1).unsqueeze(-1)  # Ensure shape [B, N, 1]
        elif pressure.dim() == 2:
            # Shape [B, N] - add dimension
            pressure = pressure.unsqueeze(-1)
        
        # Apply Fourier embedding: [B, N, 1] -> [B, N, num_wavelengths]
        B, N, _ = pressure.shape
        pressure = pressure.reshape(B * N, 1)
        p_embed = self.fourier(pressure)
        p_embed = p_embed.reshape(B, N, self.num_wavelengths)
        
        return p_embed
