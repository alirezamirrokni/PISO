from .piso import CyclePISO, CyclePISO2, GaussianPISO, GaussianPISO2
from .pzos import PZOS
from .zo_gzo import GZOHS, GZONS, ZOOG, ZOOGVR, ZOTG
from .zos import ZOS

METHODS = {
    ZOS.name: ZOS,
    ZOTG.name: ZOTG,
    ZOOG.name: ZOOG,
    ZOOGVR.name: ZOOGVR,
    GZONS.name: GZONS,
    GZOHS.name: GZOHS,
    PZOS.name: PZOS,
    GaussianPISO.name: GaussianPISO,
    CyclePISO.name: CyclePISO,
    GaussianPISO2.name: GaussianPISO2,
    CyclePISO2.name: CyclePISO2,
}

__all__ = [
    "METHODS",
    "ZOS",
    "ZOTG",
    "ZOOG",
    "ZOOGVR",
    "GZONS",
    "GZOHS",
    "PZOS",
    "GaussianPISO",
    "CyclePISO",
    "GaussianPISO2",
    "CyclePISO2",
]
