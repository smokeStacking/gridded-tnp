import itertools
import warnings
from typing import List, Optional, Tuple, Union

import einops
import numpy as np
import torch
from check_shapes import check_shapes
from torch import nn

from ..utils.grids import DownSample, UpSample
from .attention_layers import MultiHeadSelfAttentionLayer
from .teattention_layers import GriddedMultiHeadSelfTEAttentionLayer


class SWINAttentionLayer(nn.Module):
    def __init__(
        self,
        *,
        mhsa_layer: Union[
            MultiHeadSelfAttentionLayer, GriddedMultiHeadSelfTEAttentionLayer
        ],
        window_sizes: Tuple[int],
        shift_sizes: Optional[Tuple[int]] = None,
        roll_dims: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()

        self.mhsa_layer = mhsa_layer

        self.window_sizes = torch.as_tensor(window_sizes)

        if shift_sizes is not None:
            self.shift_sizes = torch.as_tensor(shift_sizes)
        else:
            self.shift_sizes = self.window_sizes // 2

        self.roll_dims = roll_dims

    @check_shapes("x: [m, ..., d]", "mask: [m, ...]", "return: [m, ..., d]")
    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn(
                "Swin Attention needs to construct its own mask, specified mask will not be used."
            )

        num_batches = x.shape[0]
        grid_shape = torch.as_tensor(x.shape[1:-1], dtype=int)
        padded_grid_shape = grid_shape + torch.as_tensor(
            tuple(
                (self.window_sizes[i] - grid_shape[i] % self.window_sizes[i]) % self.window_sizes[i]
                for i in range(len(grid_shape))
            )
        )

        # Check if window divides grid. If not, add padding + correct mask.
        if not torch.all(grid_shape % self.window_sizes == 0):
            warnings.warn(
                "Window sizes do not divide grid, adding padding tokens and correcting mask."
            )

            # Construct mask used in non-shifted attention.
            unshifted_mask = torch.ones(*grid_shape, dtype=torch.bool)
            padding_tuple = get_padding_tuple(grid_shape, self.window_sizes)
            # padding_tuple = tuple(itertools.chain(padding_tuple))  # type: ignore[arg-type]
            # flatten pad in reverse order (last dim first) and cast to ints
            flat_pad = tuple(int(v) for pair in reversed(padding_tuple) for v in pair)
            unshifted_mask = torch.nn.functional.pad(
                unshifted_mask, flat_pad, mode="constant", value=False
            )
            unshifted_mask = window_partition(
                unshifted_mask[None, ..., None], self.window_sizes
            )[0, ..., 0]
            # All tokens ignore the same tokens.
            unshifted_mask = einops.repeat(
                unshifted_mask,
                "nw ws -> m nw ws1 ws",
                m=num_batches,
                ws1=unshifted_mask.shape[1],
            )
            unshifted_mask = einops.rearrange(
                unshifted_mask,
                "m nw ws1 ws2 -> (m nw) ws1 ws2",
            )
        else:
            unshifted_mask = None

        # Adding padding.
        x = add_padding(x, self.window_sizes)
        # First no shift.
        x = window_partition(x, self.window_sizes)
        # Combine batch dimensions for efficient computation.
        x = einops.rearrange(x, "m nw ws d -> (m nw) ws d")
        x = self.mhsa_layer(x, mask=unshifted_mask)
        x = einops.rearrange(x, "(m nw) ws d -> m nw ws d", m=num_batches)
        x = window_reverse(x, self.window_sizes, padded_grid_shape)
        # Reverse the padding.
        x = remove_padding(x, grid_shape, self.window_sizes)

        # Now shift.
        shifted_x = torch.roll(
            x,
            shifts=(-self.shift_sizes).tolist(),
            dims=list(range(1, len(self.shift_sizes) + 1)),
        )
        # Add padding.
        shifted_x = add_padding(shifted_x, self.window_sizes)
        shifted_x = window_partition(shifted_x, self.window_sizes)

        # Compute attention mask for shifted windows.
        mask = swin_attention_mask(
            self.window_sizes,
            self.shift_sizes,
            grid_shape,
            self.roll_dims,
            device=x.device,
        )
        # Combine batch dimensions for efficient computation.
        mask = einops.repeat(mask, "nw ws1 ws2 -> m nw ws1 ws2", m=num_batches)
        mask = einops.rearrange(mask, "m nw ws1 ws2 -> (m nw) ws1 ws2")
        shifted_x = einops.rearrange(shifted_x, "m nw ws d -> (m nw) ws d")
        shifted_x = self.mhsa_layer(shifted_x, mask=mask)
        shifted_x = einops.rearrange(
            shifted_x, "(m nw) ws d -> m nw ws d", m=num_batches
        )
        shifted_x = window_reverse(shifted_x, self.window_sizes, padded_grid_shape)
        # Reverse the padding.
        shifted_x = remove_padding(shifted_x, grid_shape, self.window_sizes)

        # Unshift.
        x = torch.roll(
            shifted_x,
            shifts=(self.shift_sizes).tolist(),
            dims=list(range(1, len(self.shift_sizes) + 1)),
        )
        return x


class HierarchicalSWINAttentionLayer(nn.Module):
    def __init__(
        self,
        *,
        grid_shapes: Tuple[Tuple[int, ...], ...],
        swin_layers: Tuple[MultiHeadSelfAttentionLayer, ...],
    ):
        super().__init__()

        assert len(grid_shapes) == len(
            swin_layers
        ), "Number of grid sizes and layers must match."

        self.grid_shapes = grid_shapes
        self.swin_layers = nn.ModuleList(swin_layers)

        # Build the upsamplers and downsamplers.
        embed_dim = swin_layers[0].embed_dim
        self.downsamplers = nn.ModuleList(
            DownSample(embed_dim, grid_shapes[i], grid_shapes[i + 1])
            for i in range(len(grid_shapes) - 1)
        )
        self.upsamplers = nn.ModuleList(
            UpSample(embed_dim, grid_shapes[-i], grid_shapes[-(i + 1)])
            for i in range(1, len(grid_shapes))
        )

        # Build the layers that mix the original with the residual.
        self.mixing_layers = nn.ModuleList(
            nn.Linear(2 * embed_dim, embed_dim, bias=False)
            for _ in range(len(grid_shapes) - 1)
        )

    @check_shapes("x: [m, ..., d]", "mask: [m, ...]", "return: [m, ..., d]")
    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn(
                "Swin Attention needs to construct its own mask, specified mask will not be used."
            )

        # Go up through the hierarchy.
        residuals = []
        for i, swin_layer in enumerate(self.swin_layers):
            x = swin_layer(x)
            if i < len(self.downsamplers):
                residuals.append(x)
                x = self.downsamplers[i](x)

        # Now go down through the hierarchy, combining the residuals.
        for i in range(len(residuals)):
            x = self.upsamplers[i](x)
            x_mix = torch.cat((residuals[-(i + 1)], x), dim=-1)
            x = self.mixing_layers[i](x_mix)

        return x


@check_shapes(
    "x: [m, ..., d]",
    "return: [m, nw, ws, d]",
)
def window_partition(x: torch.Tensor, window_sizes: torch.Tensor):
    grid_shape = x.shape[1:-1]

    n_strings, d_strings = [f"n{i}" for i in range(len(grid_shape))], [
        f"d{i}" for i in range(len(grid_shape))
    ]
    paired = " ".join([f"({n} {d})" for n, d in zip(n_strings, d_strings)])
    reshape_pattern = (
        f"b {paired} e -> b ({' '.join(n_strings)}) ({' '.join(d_strings)}) e"
    )
    reshape_vars = dict(zip(d_strings, window_sizes))
    return einops.rearrange(x, reshape_pattern, **reshape_vars)


@check_shapes(
    "x: [m, nw, ws, d]",
    "return: [m, ..., d]",
)
def window_reverse(
    x: torch.Tensor, window_sizes: torch.Tensor, grid_shape: torch.Tensor
):
    num_windows = grid_shape // window_sizes
    n_strings, d_strings = [f"n{i}" for i in range(len(grid_shape))], [
        f"d{i}" for i in range(len(grid_shape))
    ]
    paired = " ".join([f"({n} {d})" for n, d in zip(n_strings, d_strings)])
    unreshape_pattern = (
        f"b ({' '.join(n_strings)}) ({' '.join(d_strings)}) e -> b {paired} e"
    )
    window_size_vars = dict(zip(d_strings, window_sizes))
    num_windows_vars = dict(zip(n_strings, num_windows))
    unreshape_vars = {
        **window_size_vars,
        **num_windows_vars,
    }

    return einops.rearrange(x, unreshape_pattern, **unreshape_vars)


def swin_attention_mask(
    window_sizes: torch.Tensor,
    shift_sizes: torch.Tensor,
    grid_shape: torch.Tensor,
    roll_dims: Optional[Tuple[int, ...]] = None,
    device: str = "cpu",
):
    img_mask = torch.ones((1, *grid_shape, 1), device=device)
    slices: List[Tuple[slice, ...]] = [
        (
            slice(0, -window_sizes[i]),
            slice(-window_sizes[i], -shift_sizes[i]),
            slice(-shift_sizes[i], None),
        )
        for i in range(len(grid_shape))
    ]

    if roll_dims is not None:
        for dim in roll_dims:
            slices[dim] = (
                slice(0, -window_sizes[dim]),
                slice(-window_sizes[dim], None),
            )

    cnt = 0
    for slices_ in itertools.product(*slices):
        slices_ = (slice(None), *slices_, slice(None))
        img_mask[slices_] = cnt
        cnt += 1

    # Pad to multiple of window size and assign padded patches ot a separate group.
    img_mask = add_padding(img_mask, window_sizes, value=cnt)

    # (1, num_windows, tokens_per_window).
    mask_windows = window_partition(img_mask, window_sizes).squeeze(-1)

    # (num_windows, tokens_per_window, tokens_per_window).
    attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(3)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -np.inf).masked_fill(
        attn_mask == 0, float(0.0)
    )

    return attn_mask[0]


def get_padding_tuple(
    grid_shape: torch.Tensor, window_sizes: torch.Tensor
) -> Tuple[Tuple[int, int], ...]:
    padding = tuple(
        (window_sizes[i] - grid_shape[i] % window_sizes[i]) % window_sizes[i]
        for i in range(len(grid_shape))
    )
    padding_tuple = tuple(
        (padding[i] // 2, padding[i] - padding[i] // 2) for i in range(len(grid_shape))
    )

    return padding_tuple


@check_shapes("x: [m, ..., d]", "return: [m, ..., d]")
def add_padding(
    x: torch.Tensor, window_sizes: torch.Tensor, value: float = 0.0
) -> torch.Tensor:
    padding_tuple = get_padding_tuple(x.shape[1:-1], window_sizes)
    # print(f"add_padding: x.shape[1:-1]={x.shape[1:-1]}, window_sizes={window_sizes}, padding_tuple={padding_tuple}")
    padding_tuple = tuple(itertools.chain(*reversed(padding_tuple)))  # type: ignore[arg-type]
    x = nn.functional.pad(x, (0, 0, *padding_tuple), mode="constant", value=value)
    # print(f'x shape after padding: {x.shape}')
    return x


@check_shapes("x: [m, ..., d]", "return: [m, ..., d]")
def remove_padding(
    x: torch.Tensor, grid_shape: torch.Tensor, window_sizes: torch.Tensor
) -> torch.Tensor:
    padding_tuple = get_padding_tuple(grid_shape, window_sizes)
    slices = tuple(
    slice(p[0], -p[1] if p[1] > 0 else None) for p in padding_tuple
)
    x = x[:, *slices, :]

    return x
