from .baselines import GZOHS, GZONS, VanillaPG, ZOOG, ZOOGVR, ZOTG
from .pepg import PePG
from .piso import CyclePISO, CyclePISO2, GaussianPISO, GaussianPISO2

METHODS = {
    cls.name: cls
    for cls in (
        VanillaPG,
        ZOTG,
        ZOOG,
        ZOOGVR,
        GZONS,
        GZOHS,
        GaussianPISO,
        CyclePISO,
        GaussianPISO2,
        CyclePISO2,
        PePG,
    )
}

__all__ = ["METHODS"]
