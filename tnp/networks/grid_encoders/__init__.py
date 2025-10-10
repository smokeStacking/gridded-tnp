"""
Lazy re-exports to avoid circular imports.
Only loads a submodule when an attribute is first accessed.
"""
from typing import Any

__all__ = [
    "AverageGridEncoder",
    "MultiModalGridEncoder",
    "MultiModalSingleGridEncoder",
    "BasePseudoTokenGridEncoder",
    "OOTGPseudoTokenGridEncoder",
    "PseudoTokenGridEncoder",
    "PseudoTokenGridEncoderThroughTime",
    "BasePseudoTokenTEGridEncoder",
    "OOTGPseudoTokenTEGridEncoder",
    "PseudoTokenTEGridEncoder",
    "PseudoTokenTEGridEncoderThroughTime",
    "BaseSetConv",
    "OOTGSetConv",
    "SetConv",
    "SetConvThroughTime",
]

def __getattr__(name: str) -> Any:
    if name == "AverageGridEncoder":
        from .average_grid_encoder import AverageGridEncoder
        return AverageGridEncoder
    if name in ("MultiModalGridEncoder", "MultiModalSingleGridEncoder"):
        from .mm_grid_encoders import MultiModalGridEncoder, MultiModalSingleGridEncoder
        return {"MultiModalGridEncoder": MultiModalGridEncoder,
                "MultiModalSingleGridEncoder": MultiModalSingleGridEncoder}[name]
    if name in (
        "BasePseudoTokenGridEncoder",
        "OOTGPseudoTokenGridEncoder",
        "PseudoTokenGridEncoder",
        "PseudoTokenGridEncoderThroughTime",
    ):
        from .pt_grid_encoders import (
            BasePseudoTokenGridEncoder,
            OOTGPseudoTokenGridEncoder,
            PseudoTokenGridEncoder,
            PseudoTokenGridEncoderThroughTime,
        )
        return {
            "BasePseudoTokenGridEncoder": BasePseudoTokenGridEncoder,
            "OOTGPseudoTokenGridEncoder": OOTGPseudoTokenGridEncoder,
            "PseudoTokenGridEncoder": PseudoTokenGridEncoder,
            "PseudoTokenGridEncoderThroughTime": PseudoTokenGridEncoderThroughTime,
        }[name]
    if name in (
        "BasePseudoTokenTEGridEncoder",
        "OOTGPseudoTokenTEGridEncoder",
        "PseudoTokenTEGridEncoder",
        "PseudoTokenTEGridEncoderThroughTime",
    ):
        from .pt_te_grid_encoders import (
            BasePseudoTokenTEGridEncoder,
            OOTGPseudoTokenTEGridEncoder,
            PseudoTokenTEGridEncoder,
            PseudoTokenTEGridEncoderThroughTime,
        )
        return {
            "BasePseudoTokenTEGridEncoder": BasePseudoTokenTEGridEncoder,
            "OOTGPseudoTokenTEGridEncoder": OOTGPseudoTokenTEGridEncoder,
            "PseudoTokenTEGridEncoder": PseudoTokenTEGridEncoder,
            "PseudoTokenTEGridEncoderThroughTime": PseudoTokenTEGridEncoderThroughTime,
        }[name]
    if name in ("BaseSetConv", "OOTGSetConv", "SetConv", "SetConvThroughTime"):
        from .setconv_grid_encoders import (
            BaseSetConv,
            OOTGSetConv,
            SetConv,
            SetConvThroughTime,
        )
        return {
            "BaseSetConv": BaseSetConv,
            "OOTGSetConv": OOTGSetConv,
            "SetConv": SetConv,
            "SetConvThroughTime": SetConvThroughTime,
        }[name]
    raise AttributeError(f"module {__name__} has no attribute {name}")