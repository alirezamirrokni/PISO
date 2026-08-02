from .baselines import GZOHS, GZONS, VanillaPG, ZOOG, ZOTG
from .pepg_adapter import PePGBaseline
from .piso import (
    CyclePISO,
    CyclePISO2,
    GaussianPISO,
    GaussianPISO2,
    GuidedPISO,
    GuidedPISO2,
)

METHODS = {
    cls.name: cls
    for cls in (
        VanillaPG,
        ZOTG,
        ZOOG,
        GZONS,
        GZOHS,
        GaussianPISO,
        GuidedPISO,
        CyclePISO,
        GaussianPISO2,
        GuidedPISO2,
        CyclePISO2,
        PePGBaseline,
    )
}

__all__ = ["METHODS"]
