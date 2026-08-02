from src.classification_methods.cyclepiso import CyclePISO
from src.classification_methods.cyclepiso2 import CyclePISO2
from src.classification_methods.gaussianpiso import GaussianPISO
from src.classification_methods.gaussianpiso2 import GaussianPISO2
from src.classification_methods.guidedpiso import GuidedPISO
from src.classification_methods.guidedpiso2 import GuidedPISO2
from src.classification_methods.gzo_hs import GZOHS
from src.classification_methods.gzo_ns import GZONS
from src.classification_methods.zo_og import ZOOG
from src.classification_methods.zo_ogvr import ZOOGVR
from src.classification_methods.zo_tg import ZOTG

METHODS = {
    "GZO_NS": GZONS,
    "GZO_HS": GZOHS,
    "ZO_TG": ZOTG,
    "ZO_OG": ZOOG,
    "ZO_OGVR": ZOOGVR,
    "GaussianPISO": GaussianPISO,
    "GuidedPISO": GuidedPISO,
    "CyclePISO": CyclePISO,
    "GaussianPISO2": GaussianPISO2,
    "GuidedPISO2": GuidedPISO2,
    "CyclePISO2": CyclePISO2,
}
