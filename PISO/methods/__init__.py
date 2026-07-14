from piso.methods.gzo_hs import GZOHS
from piso.methods.gzo_ns import GZONS
from piso.methods.zo_og import ZOOG
from piso.methods.zo_ogvr import ZOOGVR
from piso.methods.zo_tg import ZOTG

METHODS = {
    "GZO_NS": GZONS,
    "GZO_HS": GZOHS,
    "ZO_TG": ZOTG,
    "ZO_OG": ZOOG,
    "ZO_OGVR": ZOOGVR,
}
