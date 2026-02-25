from typing import Optional, Tuple, List, Dict, Any

import numpy as np
from scipy.signal import find_peaks, peak_widths, peak_prominences, savgol_filter

from src.spectra import Spectra


def detect_spectrum_peaks(
    spec: Spectra,
    *,
    height: Optional[float] = None,
    rel_height: float = 0.5,
    distance: Optional[int] = None,
    prominence: Optional[float] = None,
    smooth: Optional[int] = None,
    baseline_subtract: bool = False,
    invert: bool = True,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    x, y = tuple(map(np.asarray, spec.unzip()))

    if smooth is not None and isinstance(smooth, int) and smooth > 1:
        pad = smooth // 2
        y_pad = np.pad(y, pad, mode="reflect")
        kernel = np.ones(smooth) / float(smooth)
        y_smooth = np.convolve(y_pad, kernel, mode="valid")
    else:
        y_smooth = y

    if baseline_subtract:
        win = 51 if y_smooth.size > 51 else (y_smooth.size // 2 * 2 + 1)
        baseline = savgol_filter(y_smooth, win, 3)
        y_proc = y_smooth - baseline
    else:
        y_proc = y_smooth

    y_for_search = 1.0 - y_proc if invert else y_proc

    find_kwargs = {}
    if height is not None:
        find_kwargs["height"] = height
    if distance is not None:
        find_kwargs["distance"] = distance
    if prominence is not None:
        find_kwargs["prominence"] = prominence

    peaks_idx, _ = find_peaks(y_for_search, **find_kwargs)
    if peaks_idx.size == 0:
        return np.array([], dtype=int), []

    prominences, _, _ = peak_prominences(y_for_search, peaks_idx)
    widths_result = peak_widths(y_for_search, peaks_idx, rel_height)
    widths = widths_result[0]
    left_ips = widths_result[2]
    right_ips = widths_result[3]

    dx = np.diff(x)
    uniform = np.allclose(dx, dx[0]) if x.size > 1 else True

    peaks_meta: List[Dict[str, Any]] = []
    for i, idx in enumerate(peaks_idx):
        idx_int = int(round(idx))
        if uniform and x.size > 1:
            width_wl = float(widths[i] * dx[0])
        else:
            left_x = np.interp(left_ips[i], np.arange(x.size), x)
            right_x = np.interp(right_ips[i], np.arange(x.size), x)
            width_wl = float(right_x - left_x)

        peaks_meta.append(
            {
                "index": int(idx_int),
                "wl": float(x[idx_int]),
                "intens": float(y[idx_int]),
                "prominence": float(prominences[i]),
                "width_points": float(widths[i]),
                "width_wl": width_wl,
                "left_ips": float(left_ips[i]),
                "right_ips": float(right_ips[i]),
            }
        )

    return peaks_idx, peaks_meta

_periodic_table = {
    1: "H",  2: "He", 3: "Li", 4: "Be", 5: "B",  6: "C",  7: "N",  8: "O",  9: "F",  10: "Ne",
    11: "Na",12: "Mg",13: "Al",14: "Si",15: "P", 16: "S", 17: "Cl",18: "Ar",19: "K",  20: "Ca",
    21: "Sc",22: "Ti",23: "V", 24: "Cr",25: "Mn",26: "Fe",27: "Co",28: "Ni",29: "Cu",30: "Zn",
    31: "Ga",32: "Ge",33: "As",34: "Se",35: "Br",36: "Kr",37: "Rb",38: "Sr",39: "Y",  40: "Zr",
    41: "Nb",42: "Mo",43: "Tc",44: "Ru",45: "Rh",46: "Pd",47: "Ag",48: "Cd",49: "In",50: "Sn",
    51: "Sb",52: "Te",53: "I", 54: "Xe",55: "Cs",56: "Ba",57: "La",58: "Ce",59: "Pr",60: "Nd",
    61: "Pm",62: "Sm",63: "Eu",64: "Gd",65: "Tb",66: "Dy",67: "Ho",68: "Er",69: "Tm",70: "Yb",
    71: "Lu",72: "Hf",73: "Ta",74: "W", 75: "Re",76: "Os",77: "Ir",78: "Pt",79: "Au",80: "Hg",
    81: "Tl",82: "Pb",83: "Bi",84: "Po",85: "At",86: "Rn",87: "Fr",88: "Ra",89: "Ac",90: "Th",
    91: "Pa",92: "U", 93: "Np",94: "Pu",95: "Am",96: "Cm",97: "Bk",98: "Cf",99: "Es",100: "Fm",
    101: "Md",102: "No",103: "Lr",104: "Rf",105: "Db",106: "Sg",107: "Bh",108: "Hs",109: "Mt",
    110: "Ds",111: "Rg",112: "Cn",113: "Nh",114: "Fl",115: "Mc",116: "Lv",117: "Ts",118: "Og"
}

def get_element_symbol(Z: int) -> Optional[str]:
    return _periodic_table.get(Z)