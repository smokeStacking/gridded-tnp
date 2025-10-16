from collections.abc import Iterable
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from ..utils.batch import compress_batch_dimensions
from .embeddings import Embedding


class MLP(nn.Module):
    """MLP.

    Args:
        in_dim (int): Input dimensionality.
        out_dim (int): Output dimensionality.
        layers (tuple[int, ...], optional): Width of every hidden layer.
        num_layers (int, optional): Number of hidden layers.
        width (int, optional): Width of the hidden layers
        nonlinearity (function, optional): Nonlinearity.
        dtype (dtype, optional): Data type.

    Attributes:
        net (object): MLP, but which expects a different data format.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        layers: Optional[Tuple[int, ...]] = None,
        num_layers: Optional[int] = None,
        width: Optional[int] = None,
        nonlinearity: Optional[nn.Module] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        if layers is None:
            # Check that one of the two specifications is given.
            assert (
                num_layers is not None and width is not None
            ), "Must specify either `layers` or `num_layers` and `width`."
            layers = (width,) * num_layers

        # Default to ReLUs.
        if nonlinearity is None:
            nonlinearity = nn.ReLU()

        # Build layers.
        if len(layers) == 0:
            self.net = nn.Linear(in_dim, out_dim, dtype=dtype)
        else:
            net = [nn.Linear(in_dim, layers[0], dtype=dtype)]
            for i in range(1, len(layers)):
                net.append(nonlinearity)
                net.append(nn.Linear(layers[i - 1], layers[i], dtype=dtype))
            net.append(nonlinearity)
            net.append(nn.Linear(layers[-1], out_dim, dtype=dtype))
            self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, uncompress = compress_batch_dimensions(x, 1)
        x = self.net(x)
        x = uncompress(x)
        return x


class MLPWithEmbedding(MLP):
    def __init__(
        self,
        embeddings: Union[Embedding, Tuple[Embedding, ...]],
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not isinstance(embeddings, Iterable):
            embeddings = (embeddings,)

        self.embeddings = nn.ModuleList(embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Embed the input.
        embedded_dims: List[int] = []
        embeddings: List[torch.Tensor] = []
        for embedding in self.embeddings:
            embedded_dims.extend(embedding.active_dims)
            embeddings.append(embedding(x[..., embedding.active_dims]))

        embedded_dims = [
            dim if dim >= 0 else x.shape[-1] + dim for dim in embedded_dims
        ]
        non_embedded_dims = [
            dim for dim in range(x.shape[-1]) if dim not in embedded_dims
        ]
        x_non_embedded = x[..., non_embedded_dims]
        x = torch.cat((x_non_embedded, *embeddings), dim=-1)
        out = super().forward(x)
        return out


class PressureConditionedMLP(nn.Module):
    """
    MLP that conditions on pressure levels via pre-computed Fourier embeddings.
    
    Two modes:
    1. Cached mode: Pressure embeddings pre-computed and stored as buffer [H*W, L, F]
       - Memory efficient: ~88 MB for 360x180x34 grid
       - Zero computation: just indexing and broadcasting
       
    2. Dynamic mode (fallback): Computes embeddings on-the-fly if cache not provided
       - Flexible but slower
    
    Flow (cached mode):
        Temperature [B, N, L]
             ↓
        [B, N, L, 1]
             └──────────concat──────────┐
                                        ↓
        Cached Embeddings [H*W, L, F]  [B, N, L, 1+F]
             ↓ expand to [B, N, L, F]    ↓
                                   Flatten: [B, N, L*(1+F)]
                                        ↓
                                   MLP: L*(1+F) → out_dim
                                        ↓
                                   [B, N, out_dim]
    """
    def __init__(
        self,
        value_dim: int,
        out_dim: int,
        num_layers: int,
        width: int,
        pressure_embeddings: Optional[torch.Tensor] = None,
        pressure_encoder: Optional[nn.Module] = None,
        nonlinearity: Optional[nn.Module] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,  # Absorb any extra config params
    ):
        super().__init__()
        
        # Determine number of pressure features
        if pressure_embeddings is not None:
            # Cache mode: use pre-computed embeddings
            # Expected shape: [H*W, L, F]
            if pressure_embeddings.dim() != 3:
                raise ValueError(
                    f"pressure_embeddings must be [H*W, L, F], got shape {pressure_embeddings.shape}"
                )
            self.register_buffer('pressure_embeddings', pressure_embeddings)
            self.pressure_encoder = None
            HW, L, F = pressure_embeddings.shape
            self.num_pressure_features = F
            self.spatial_size = HW
            
        elif pressure_encoder is not None:
            # Dynamic mode: compute on-the-fly
            self.pressure_embeddings = None
            self.pressure_encoder = pressure_encoder
            self.num_pressure_features = pressure_encoder.num_wavelengths
            self.spatial_size = None
        else:
            raise ValueError(
                "Must provide either pressure_embeddings (cached) or pressure_encoder (dynamic)"
            )
        
        # Input: L × (1 value + F pressure features)
        mlp_in_dim = value_dim * (1 + self.num_pressure_features)
        
        self.mlp = MLP(
            in_dim=mlp_in_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            width=width,
            nonlinearity=nonlinearity,
            dtype=dtype,
        )
    
    def forward(
        self, 
        y: torch.Tensor, 
        pressure: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            y: [B, N, L] variable values where N = T*H*W
            pressure: Optional[B, N, L] pressure values (only needed if no cache)
        
        Returns:
            [B, N, out_dim] encoded values conditioned on pressure
        """
        B, N, L = y.shape
        
        # Get pressure embeddings (cached or dynamic)
        if self.pressure_embeddings is not None:
            # Use cached embeddings: [H*W, L, F]
            HW, L_cached, F = self.pressure_embeddings.shape
            
            if L != L_cached:
                raise ValueError(
                    f"Level dimension mismatch: y has {L} levels, "
                    f"cached embeddings have {L_cached} levels"
                )
            
            if N % HW != 0:
                raise ValueError(
                    f"Spatial dimension mismatch: N={N} must be multiple of H*W={HW}"
                )
            
            # Expand cached embeddings to match batch and time dimensions
            T = N // HW
            # [H*W, L, F] -> [1, 1, H*W, L, F] -> [B, T, H*W, L, F] -> [B, N, L, F]
            p_embed = self.pressure_embeddings.unsqueeze(0).unsqueeze(0)
            p_embed = p_embed.expand(B, T, -1, -1, -1)
            p_embed = p_embed.reshape(B, N, L, F)
            # print(f"PressureConditionedMLP: cached embeddings shape: {p_embed.shape}")
            
        else:
            # Dynamic computation using encoder
            if pressure is None:
                raise ValueError(
                    "pressure must be provided when using dynamic mode (no cache)"
                )
            p_embed = self.pressure_encoder(pressure)  # [B, N, L, F]
        
        # Concatenate value with pressure features
        y_expanded = y.unsqueeze(-1)  # [B, N, L, 1]
        fused = torch.cat([y_expanded, p_embed], dim=-1)  # [B, N, L, 1+F]
        # Flatten and project: [B, N, L*(1+F)] -> [B, N, out_dim]
        return self.mlp(fused.reshape(B, N, -1))
