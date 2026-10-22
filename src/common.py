import hashlib
import json
import os
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

plt.rcParams.update({
    "font.size": 14.0, "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

STYLES = {
    "GZO_NS": {"color": "blue", "marker": "o", "linestyle": "-"},
    "GZO_HS": {"color": "cyan", "marker": "s", "linestyle": "-"},
    "ZO_TG": {"color": "green", "marker": "^", "linestyle": "-"},
    "ZO_OGVR": {"color": "black", "marker": "*", "linestyle": "-"},
    "GaussianPISO": {"color": "purple", "marker": "X", "linestyle": "-"},
    "CyclePISO": {"color": "magenta", "marker": "v", "linestyle": "-"},
    "GaussianPISO2": {"color": "tab:brown", "marker": "h", "linestyle": "-"},
    "CyclePISO2": {"color": "tab:pink", "marker": "<", "linestyle": "-"},
    "PZOS": {"color": "darkorange", "marker": "P", "linestyle": "-"},
    "ZOS": {"color": "tab:olive", "marker": ">", "linestyle": "--"},
    "PePG": {"color": "darkorange", "marker": "P", "linestyle": "-"},
    "VanillaPG": {"color": "tab:olive", "marker": ">", "linestyle": "--"},
}
LABELS = {
    "GZO_NS": "GZO-NS", "GZO_HS": "GZO-HS", "ZO_TG": "ZO-TG", "ZO_OGVR": "ZO-OGVR",
    "GaussianPISO2": "GaussianPISO$^2$", "CyclePISO2": "CyclePISO$^2$", "VanillaPG": "Vanilla PG",
}


def label(method):
    return LABELS.get(method, method)


def hash_seed(text, nbytes=4):
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:nbytes], "little")


def short_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_config(config, path):
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


class RunStore:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def load(self, key):
        path = self.folder / f"{key}.pkl"
        if not path.exists():
            return None
        with path.open("rb") as handle:
            return pickle.load(handle)

    def save(self, key, value):
        path = self.folder / f"{key}.pkl"
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            pickle.dump(value, handle)
        os.replace(temporary, path)


def run_cached(store, key, rng, function):
    saved = store.load(key)
    if saved is None:
        result = function()
        store.save(key, (result, rng.get_state()))
        return result
    rng.set_state(saved[1])
    return saved[0]


def style_axis(axis, label_size=17.0, tick_size=13.0, title_size=17.0):
    axis.xaxis.label.set_size(label_size)
    axis.yaxis.label.set_size(label_size)
    axis.title.set_size(title_size)
    for tick in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
        tick.set_fontsize(tick_size)
        tick.set_fontweight("bold")


def legend_below(figure, handles, labels, size=12.0, y=0.018):
    if handles:
        figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y), ncol=len(handles),
                      frameon=False, prop={"size": size, "weight": "bold"}, handlelength=1.25,
                      columnspacing=0.48, handletextpad=0.28)


def save_figure(figure, stem, dpi=300, tight=True):
    options = {"bbox_inches": "tight"} if tight else {}
    figure.savefig(f"{stem}.png", dpi=dpi, **options)
    figure.savefig(f"{stem}.pdf", **options)
    plt.close(figure)
