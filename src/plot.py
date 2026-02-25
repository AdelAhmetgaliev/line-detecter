from typing import Tuple, Iterable, Optional, Any

import numpy as np
import matplotlib.pyplot as plt

from src.spectra import Spectra

FONTSIZE: int = 14
COLOR: str = "k"


def save_figure(
    fig,
    path: str,
    *,
    dpi: int = 300,
    transparent: bool = False,
    bbox_inches: str = "tight",
    pad_inches: float = 0.02,
    quality: int | None = None,
) -> None:
    save_kwargs = {
        "dpi": dpi,
        "transparent": transparent,
        "bbox_inches": bbox_inches,
        "pad_inches": pad_inches,
        "facecolor": fig.get_facecolor(),
        "edgecolor": "none",
    }

    ext = path.split(".")[-1].lower()
    if ext in {"png", "jpg", "jpeg", "tif", "tiff"}:
        if ext in {"jpg", "jpeg"} and quality is not None:
            save_kwargs["quality"] = max(1, min(95, int(quality)))
    else:
        save_kwargs.pop("quality", None)

    fig.savefig(path, **save_kwargs)


def plot_spectra(spectra: Spectra, *, show: bool = True) -> Tuple:
    wl_list, intens_list = spectra.unzip()
    wl = np.asarray(wl_list, dtype=float)
    inten = np.asarray(intens_list, dtype=float)

    if wl.size == 0:
        raise ValueError("Spectrum is empty")

    plt.rcParams.update(
        {
            "font.size": FONTSIZE,
            "font.family": "sans-serif",
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
        }
    )

    fig, ax = plt.subplots(1, 1, figsize=(18, 9))

    ax.plot(
        wl,
        inten,
        color=COLOR,
        linewidth=1.6,
        label="Спектр звезды",
    )

    ax.set_xlabel("Длина волны, A", fontsize=FONTSIZE)
    ax.set_ylabel("Относительная интенсивность", fontsize=FONTSIZE)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.tick_params(which="both", top=True, right=True, labelsize=FONTSIZE)
    ax.minorticks_on()

    plt.tight_layout()
    if show:
        plt.show()

    return fig, ax


def draw_peak_lines(
    fig: Any,
    ax: Any,
    peaks_meta: Iterable[dict],
    *,
    color: str = "red",
    linestyle: str = "--",
    linewidth: float = 0.5,
    alpha: float = 0.3,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    label_fmt: Optional[str] = None,
) -> None:
    xs = []
    for meta in peaks_meta:
        if "wl" in meta:
            xs.append(float(meta["wl"]))
        elif "index" in meta and hasattr(ax, "lines"):
            xs.append(float(meta["index"]))
        else:
            continue

    if not xs:
        return

    y0, y1 = ax.get_ylim()
    if y_min is None:
        y_min = y0
    if y_max is None:
        y_max = y1

    for i, xval in enumerate(xs):
        ax.vlines(
            xval,
            y_min,
            y_max,
            colors=color,
            linestyles=linestyle,
            linewidth=linewidth,
            alpha=alpha,
        )
        if label_fmt is not None:
            meta = list(peaks_meta)[i]
            try:
                label = label_fmt.format(**meta)
            except Exception:
                label = str(xval)
            ax.text(
                xval,
                y_max,
                label,
                rotation=90,
                va="bottom",
                ha="center",
                color=color,
                alpha=alpha,
                fontsize=8,
            )

    fig.canvas.draw_idle()


def label_element_on_line(
    ax: Any,
    wl: float,
    label: str,
    *,
    y: Optional[float] = None,
    offset_y: float = -0.5,
    offset_x: float = -0.05,
    color: str = "red",
    fontsize: int = 12,
    rotation: float = 90,
    ha: str = "center",
    va: str = "bottom",
) -> None:
    ymin, ymax = ax.get_ylim()
    if y is None:
        y_text = ymax + offset_y * (ymax - ymin)
    else:
        y_text = y

    x_text = wl + offset_x

    ax.axvline(wl, color=color, linestyle="--", linewidth=0.8, alpha=0.8)
    ax.text(
        x_text,
        y_text,
        label,
        color=color,
        fontsize=fontsize,
        rotation=rotation,
        ha=ha,
        va=va,
    )
    ax.figure.canvas.draw_idle()
