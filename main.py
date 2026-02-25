import csv
import logging
from typing import Iterable, List, Sequence, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

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


# --- Configuration -----------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    project_root: Path = Path(__file__).parent
    data_dir: Path = project_root / "data"
    output_dir: Path = project_root / "output"
    spectra_file: str = "spectra.dat"
    lines_file: str = "lines.dat"
    spectra_bounds: Tuple[float, float] = (5250.0, 5300.0)
    peak_height: float = 0.1
    strength_reference_temp: float = 7800.0
    strength_threshold: float = 0.35
    save_plots: bool = False
    show_plots: bool = False
    figure_name: str = "spectra_adel.png"
    figure_dpi: int = 500
    result_csv: str = "result.csv"


CFG = Config()

# --- Logging -----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# --- I/O Helpers -------------------------------------------------------------------
def ensure_paths(cfg: Config) -> None:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)


def load_inputs(cfg: Config) -> Tuple[Path, Path]:
    spectra_path = cfg.data_dir / cfg.spectra_file
    lines_path = cfg.data_dir / cfg.lines_file

    if not spectra_path.exists():
        raise FileNotFoundError(f"spectra file not found: {spectra_path}")
    if not lines_path.exists():
        raise FileNotFoundError(f"lines file not found: {lines_path}")

    return spectra_path, lines_path


# --- Analysis ----------------------------------------------------------------------
def select_candidates(
    cand_list: Sequence[SpectralLine],
    temp: float,
    threshold: float,
) -> List[Tuple[SpectralLine, float]]:
    """Return candidate lines with strength within threshold of the best."""
    strengths: List[Tuple[SpectralLine, float]] = [
        (line, calc_line_strength(line, temp)) for line in cand_list
    ]
    if not strengths:
        return []
    strengths.sort(key=lambda x: x[1], reverse=True)
    max_s = strengths[0][1]
    selected = [t for t in strengths if abs(t[1] - max_s) < threshold]
    return selected


def format_label(candidates: Sequence[Tuple[SpectralLine, float]]) -> str:
    if not candidates:
        return "no match"
    parts = [
        f"{get_element_symbol(line.element)}{'I' * line.ion_stage}"
        for line, _ in candidates
    ]
    return " + ".join(parts)


# --- CSV Output --------------------------------------------------------------------
CSV_HEADERS = ["λ изм, Å", "λ теор, Å", "элемент", "стадия ионизации", "S", "I_λ"]


def write_results_csv(
    path: Path,
    rows: Iterable[Tuple[float, Sequence[Tuple[SpectralLine, float]], Optional[float]]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)
        for lambda_izm, candidates, I_lambda in rows:
            first = True
            I_val = I_lambda if I_lambda is not None else ""
            for line_obj, S in candidates:
                writer.writerow(
                    [
                        f"{lambda_izm:.2f}" if first else "",
                        line_obj.laboratory_wavelength,
                        get_element_symbol(line_obj.element),
                        "I" * line_obj.ion_stage,
                        f"{S:.6g}",
                        I_val,
                    ]
                )
                first = False


# --- Main flow --------------------------------------------------------------------
def analyze_and_plot(
    cfg: Config,
) -> List[Tuple[float, List[Tuple[SpectralLine, float]], Optional[float]]]:
    spectra_path, lines_path = load_inputs(cfg)

    spec = Spectra.from_file(spectra_path).filter_by_wavelength(*cfg.spectra_bounds)
    lines = parse_linefile(lines_path)  # type: ignore

    logger.info(
        "Loaded spectrum (points=%d) and lines (count=%d)",
        len(spec.wavelengths()),
        len(lines),
    )

    _, peak_metas = detect_spectrum_peaks(spec, height=cfg.peak_height)

    fig, ax = plot_spectra(spec, show=False)
    draw_peak_lines(fig, ax, peak_metas)

    results: List[Tuple[float, List[Tuple[SpectralLine, float]], Optional[float]]] = []

    for peak in peak_metas:
        obs_wl = peak["wl"]
        cand_list = find_candidates_for_observed(obs_wl, lines)
        if not cand_list:
            label = "no match"
            label_element_on_line(ax, obs_wl, label, y=peak.get("intens", 0.0) / 1.07)
            results.append((obs_wl, [], peak.get("intens")))
            continue

        selected = select_candidates(
            cand_list, cfg.strength_reference_temp, cfg.strength_threshold
        )
        if not selected:
            label = "no match"
        else:
            label = format_label(selected)

        label_element_on_line(ax, obs_wl, label, y=peak.get("intens", 0.0) / 1.07)
        results.append((obs_wl, selected, peak.get("intens")))

    if cfg.save_plots:
        out_fig = cfg.output_dir / cfg.figure_name
        save_figure(fig, str(out_fig), dpi=cfg.figure_dpi)
        logger.info("Saved figure to %s", out_fig)

    if cfg.show_plots:
        plt.show()

    plt.close(fig)
    return results


def main() -> None:
    ensure_paths(CFG)
    results = analyze_and_plot(CFG)
    csv_path = CFG.output_dir / CFG.result_csv
    write_results_csv(csv_path, results)
    logger.info("Wrote CSV results to %s", csv_path)


if __name__ == "__main__":
    main()
