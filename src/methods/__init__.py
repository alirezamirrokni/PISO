from src.methods.gzo_hs import GZOHS
from src.methods.gzo_ns import GZONS
from src.methods.piso import PISO
from src.methods.piso_m import PISOM
from src.methods.zo_og import ZOOG
from src.methods.zo_ogvr import ZOOGVR
from src.methods.zo_tg import ZOTG

METHODS = {
    "GZO_NS": GZONS,
    "GZO_HS": GZOHS,
    "ZO_TG": ZOTG,
    "ZO_OG": ZOOG,
    "ZO_OGVR": ZOOGVR,
    "PISO": PISO,
    "PISO_M": PISOM,
}
