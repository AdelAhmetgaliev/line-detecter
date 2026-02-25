import os
import csv

from pathlib import Path
# from pprint import pprint

import matplotlib.pyplot as plt

from src.spectra import Spectra
from src.plot import save_figure, plot_spectra, draw_peak_lines, label_element_on_line
from src.utils import detect_spectrum_peaks, get_element_symbol
from src.lines import (
    SpectralLine,
    parse_linefile,
    find_candidates_for_observed,
    calc_line_strength,
)

DATA_DIR: str = os.path.join(os.path.dirname(__file__), "data")
SPECTRA_BOUNDS: tuple[float, float] = (5250.0, 5300.0)
# SPECTRA_BOUNDS: tuple[float, float] = (5295.0, 5350.0)

SHOW_PLOTS: bool = False
SAVE_PLOTS: bool = False


def main() -> None:
    path_to_spectra = Path(os.fspath(DATA_DIR)) / "spectra.dat"
    path_to_lines = Path(os.fspath(DATA_DIR)) / "lines.dat"

    spec: Spectra = Spectra.from_file(path_to_spectra).filter_by_wavelength(
        *SPECTRA_BOUNDS
    )
    lines: list[SpectralLine] = parse_linefile(path_to_lines)  # type: ignore

    _, peak_metas = detect_spectrum_peaks(spec, height=0.1)

    fig, ax = plot_spectra(spec, show=False)
    draw_peak_lines(fig, ax, peak_metas)

    data = []
    for peak in peak_metas:
        obs_wl = peak["wl"]
        cand_list = find_candidates_for_observed(obs_wl, lines)  # type: ignore

        line_strength_list: list[tuple[SpectralLine, float]] = []
        for line in cand_list:
            strength = calc_line_strength(line, 7800.0)
            line_strength_list.append((line, strength))

        line_strength_list.sort(key=lambda tup: tup[1], reverse=True)
        max_strength = line_strength_list[0][1]

        new_cand_list: list[tuple] = []
        for tup in line_strength_list:
            if abs(tup[1] - max_strength) < 0.35:
                new_cand_list.append(tup)

        label_str = ""
        idx = 0
        for tup in new_cand_list:
            if idx < len(new_cand_list) - 1:
                label_str += (
                    f"{get_element_symbol(tup[0].element)}{'I' * tup[0].ion_stage} + "
                )
            idx += 1
        label_str += f"{get_element_symbol(new_cand_list[-1][0].element)}{'I' * new_cand_list[-1][0].ion_stage}"
        label_element_on_line(ax, obs_wl, label_str, y=peak["intens"] / 1.07)
        data.append([obs_wl, new_cand_list, peak["intens"]])

    if SAVE_PLOTS:
        save_figure(fig, "spectra_adel.png", dpi=500)
    if SHOW_PLOTS:
        plt.show()

    with open(file="result.csv", mode="w+", encoding="utf-8", newline="") as res_file:
        writer = csv.writer(res_file)
        writer.writerow(
            ["λ изм, Å", "λ теор, Å", "элемент", "стадия ионизации", "S", "I_λ"]
        )
        for entry in data:
            lambda_izm = entry[0]
            lines: list[tuple] = entry[1]
            I_lambda = entry[2] if len(entry) > 2 else ""
            first = True
            for line_obj, S in lines:
                writer.writerow(
                    [
                        f"{lambda_izm:.2f}" if first else "",
                        line_obj.laboratory_wavelength,
                        get_element_symbol(line_obj.element),
                        "I" * line_obj.ion_stage,
                        S,
                        I_lambda,
                    ]
                )
                first = False


if __name__ == "__main__":
    main()
