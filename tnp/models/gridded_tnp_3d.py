import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.grid_encoders import (
    BasePseudoTokenGridEncoder,
    MultiModalGridEncoder,
    OOTGPseudoTokenGridEncoder,
    OOTGSetConv,
    SetConv,
)
from ..networks.transformer import GriddedTransformerEncoder
from ..utils.helpers import preprocess_observations
from .base import (
    ConditionalNeuralProcess,
    MultiModalConditionalNeuralProcess,
    OOTGConditionalNeuralProcess,
)
from .tnp import TNPDecoder
from ..networks.mlp import MLP
from ipdb import set_trace


class MultiModalGriddedTNPEncoder3D(nn.Module):
    def __init__(
        self,
        transformer_encoder: GriddedTransformerEncoder,
        grid_encoder: MultiModalGridEncoder,
        xy_encoder: Union[nn.Module, Dict[str, nn.Module]],
        zt_encoder: nn.Module,
        mode_names: List[str],
        x_encoder: nn.Module = nn.Identity(),
        y_encoder: Union[nn.Module, Dict[str, nn.Module]] = nn.Identity(),
        x_grid_dims: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()

        self.transformer_encoder = transformer_encoder
        self.grid_encoder = grid_encoder
        self.mode_names = mode_names

        # Shared x_encoder.
        self.x_encoder = x_encoder

        self.zt_encoder = zt_encoder

        if isinstance(xy_encoder, dict):
            # For dict input, we need to know what the default should be
            # We'll require a special '_default' key to specify the fallback
            default_encoder = xy_encoder.pop('_default', None)
            if default_encoder is None:
                raise ValueError("When xy_encoder is a dict, must include '_default' key for unspecified modes")
            
            self.xy_encoder = nn.ModuleDict({
                mode: copy.deepcopy(xy_encoder.get(mode, default_encoder)) 
                for mode in mode_names
            })
        else:
            # Original behavior: copy single encoder for all modes
            self.xy_encoder = nn.ModuleDict(
                {mode: copy.deepcopy(xy_encoder) for mode in mode_names}
            )
    
        # If y_encoder dict has a mapping for a mode, use it, otherwise use the identity
        if isinstance(y_encoder, dict):
            self.y_encoder = nn.ModuleDict({
                mode: y_encoder.get(mode, nn.Identity()) for mode in mode_names
            })
        else:
            self.y_encoder = nn.ModuleDict(
                {mode: copy.deepcopy(y_encoder) for mode in mode_names}
            )

        self.x_grid_dims = x_grid_dims

    def forward(
        self,
        xc: Dict[str, torch.Tensor],
        yc: Dict[str, torch.Tensor],
        xt: torch.Tensor,
        time_grid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        print(f"G-TNP Encoder xt shape: {xt.shape}, time_grid shape: {time_grid.shape}")

        # Use separate encoders for each modality.
        xc_list = list(xc.values()) # Get all mode coordinate tensors
        yc_list = list(yc.values())
        nc_list = [xcm.shape[-2] for xcm in xc_list] # Track sizes per mode
        x = torch.cat(xc_list, dim=1) # Concatenate along point dimension
        x = torch.cat((x, xt), dim=1) # Add target coords
        x_encoded = self.x_encoder(x) # Process all coordinates together

        # Split x_encoded into xc_list_encoded and xt_encoded.
        xc_list_encoded = x_encoded[:, : sum(nc_list)].split(nc_list, dim=1)
        xt_encoded = x_encoded[:, sum(nc_list) :]

        # Encode xt.
        zt = self.zt_encoder(xt_encoded)

        # Encode y for each modality.
        yc_encoded_list = []

        if self.y_encoder is not None:
            for mod, yc_ in zip(yc, yc_list):
                if yc_.dim() > 3: #3d variables need to be flattened: {B, xc, yc, 1} -> {B, xc, yc}
                    yc_ = yc_.squeeze(dim=-1)
                yc_encoded_list.append(self.y_encoder[mod](yc_))
        else:
            yc_encoded_list = yc_list

        # Combine x and y encodings for each modality
        zc: Dict[str, torch.Tensor] = {}
        for mod, xc_encoded, yc_encoded in zip(
            yc.keys(), xc_list_encoded, yc_encoded_list
        ):
            zc[mod] = torch.cat((xc_encoded, yc_encoded), dim=-1)
            zc[mod] = self.xy_encoder[mod](zc[mod])

        if self.x_grid_dims is not None:
            xc = {mod: xc[mod][..., self.x_grid_dims] for mod in xc.keys()}
            xt = xt[..., self.x_grid_dims]

        # Encode to grid using original xc.
        if time_grid is not None:
            xc_grid, zc_grid = self.grid_encoder(xc, zc, time_grid)
        else:
            xc_grid, zc_grid = self.grid_encoder(xc, zc)

        # Apply transformer encoder
        zt = self.transformer_encoder(xc_grid, zc_grid, xt, zt)
        return zt




class MultiModalGriddedTNP3D(MultiModalConditionalNeuralProcess):
    
    def __init__(
        self,
        encoder: MultiModalGriddedTNPEncoder3D,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
    
    #special check to allow different shapes for xc and yc
    @check_shapes(
        "xc.values(): [m, ..., dx]",
        "yc.values(): [m, ..., ...]",
        "xt: [m, nt, dx]",
        "time_grid: [m, t_grid]",
    )
    def forward(
        self,
        xc: Dict[str, torch.Tensor],
        yc: Dict[str, torch.Tensor],
        xt: torch.Tensor,
        time_grid: Optional[torch.Tensor] = None,
    ) -> torch.distributions.Distribution:
        return self.likelihood(self.decoder(self.encoder(xc, yc, xt, time_grid), xt))
