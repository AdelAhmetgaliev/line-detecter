from dataclasses import dataclass
from typing import List, Iterable, Union
from pathlib import Path


@dataclass
class SpectralLine:
    laboratory_wavelength: float
    element: int
    ion_stage: int
    log_10_A: float
    log_10_gf: float
    lower_level_E: float
    ionization_energy_lower: float
    ionization_energy_upper: float

    def matches_wavelength(self, obs_lambda: float, tol: float) -> bool:
        return abs(self.laboratory_wavelength - obs_lambda) <= tol

    def doppler_shifted(self, rv_km_s: float) -> float:
        c_km_s = 299792.458
        return self.laboratory_wavelength * (1 + rv_km_s / c_km_s)


def _parse_float_or_zero(s: str) -> float:
    return 0.0 if s == "--" else float(s)

def _parse_int_or_zero(s: str) -> int:
    return 0 if s == "--" else int(s)

def parse_linefile(path: Union[str, Path]) -> List[SpectralLine]:
    cols_expected = 8
    entries: List[SpectralLine] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) < cols_expected:
                continue
            try:
                lam = _parse_float_or_zero(parts[0])
                element = _parse_int_or_zero(parts[1])
                ion_stage = _parse_int_or_zero(parts[2])
                logA = _parse_float_or_zero(parts[3])
                loggf = _parse_float_or_zero(parts[4])
                Ei = _parse_float_or_zero(parts[5])
                Eion_l = _parse_float_or_zero(parts[6])
                Eion_u = _parse_float_or_zero(parts[7])
            except (ValueError, IndexError):
                continue
            entries.append(
                SpectralLine(
                    laboratory_wavelength=lam,
                    element=element,
                    ion_stage=ion_stage,
                    log_10_A=logA,
                    log_10_gf=loggf,
                    lower_level_E=Ei,
                    ionization_energy_lower=Eion_l,
                    ionization_energy_upper=Eion_u,
                )
            )
    return entries


def find_candidates_for_observed(
    observed_lambda: float, lines: Iterable[SpectralLine], tol: float = 0.2
) -> List[SpectralLine]:
    return [ln for ln in lines if ln.matches_wavelength(observed_lambda, tol)]


def _ku_from_Eu(Eu: float) -> float:
    if Eu is None:
        return 0.0

    if 5.0 <= Eu < 6.0:
        return -5.0
    if 6.0 <= Eu < 7.0:
        return -4.0
    if 7.0 <= Eu < 7.5:
        return -3.5
    if 7.5 <= Eu < 8.0:
        return -3.0
    if 8.0 <= Eu < 9.0:
        return -2.5
    if 9.0 <= Eu < 10.0:
        return -1.5
    if 10.0 <= Eu < 11.0:
        return -1.0
    if 11.0 <= Eu < 13.0:
        return -0.5
    if Eu >= 13.0:
        return 0.0
    return 0.0


def _kl_from_El(El: float) -> float:
    if El is None:
        return 0.0

    if 5.0 <= El < 11.5:
        return 0.0
    if 11.5 <= El < 13.0:
        return -0.5
    if 13.0 <= El < 15.0:
        return -1.0
    if 15.0 <= El < 18.0:
        return -1.5
    if El >= 18.0:
        return -2.0
    return 0.0


def _lg_f_from_line(line: SpectralLine) -> float:
    Eu = line.ionization_energy_upper
    El = line.ionization_energy_lower
    ku = _ku_from_Eu(Eu)
    kl = _kl_from_El(El)
    return ku + kl


def calc_line_strength(line: SpectralLine, T_kelvin: float) -> float:
    k_B_ev_per_K = 8.617333262145e-5
    lgf_table = _lg_f_from_line(line)
    return (
        line.log_10_A
        + line.log_10_gf
        + lgf_table
        - (1.0 / 2.3) * (line.lower_level_E / (k_B_ev_per_K * T_kelvin))
    )
