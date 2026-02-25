import os
import numpy as np

from pathlib import Path
from typing import Self, Iterable
from scipy.signal import savgol_filter

SpectraPoint = tuple[float, float]
PathLikeStr = os.PathLike[str] | str | Path


class Spectra:
    __slots__ = ("data",)
    data: list[SpectraPoint]

    def __init__(self, data: Iterable[SpectraPoint] | None = None) -> None:
        self.data = list(data) if data is not None else []

    @classmethod
    def from_file(cls, file_path: PathLikeStr, *, encoding: str = "utf-8") -> Self:
        p = Path(os.fspath(file_path))
        data: list[SpectraPoint] = []
        with p.open("r", encoding=encoding) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) != 2:
                    continue
                try:
                    wl, inten = float(parts[0]), float(parts[1])
                except ValueError:
                    continue

                data.append((wl, inten))

        return cls(data)

    @classmethod
    def from_arrays(
        cls, wavelengths: Iterable[float], intensities: Iterable[float]
    ) -> Self:
        return cls(list(zip(wavelengths, intensities)))

    def wavelengths(self) -> list[float]:
        return [wl for wl, _ in self.data]

    def intensities(self) -> list[float]:
        return [inten for _, inten in self.data]

    def unzip(self) -> tuple[list[float], list[float]]:
        if not self.data:
            return [], []

        wl, inten = zip(*self.data)

        return list(wl), list(inten)

    def normalize_intensities(self, *, clip: bool = True) -> Self:
        wl_list, inten_list = self.unzip()
        if not inten_list:
            return self.__class__([])

        min_int = min(inten_list)
        max_int = max(inten_list)
        denom = max_int - min_int or 1.0
        normalized = [
            (wl, (inten - min_int) / denom) for wl, inten in zip(wl_list, inten_list)
        ]

        return self.__class__(normalized)

    def apply_savgol(self, window_length: int, poly_order: int) -> Self:
        wl_list, inten_list = self.unzip()
        n = len(inten_list)

        if n == 0:
            return self.__class__([])
        if window_length <= 0 or poly_order < 0:
            raise ValueError("window_length must be > 0 and poly_order >= 0")
        if window_length > n:
            window_length = n if n % 2 == 1 else n - 1
        if window_length % 2 == 0:
            window_length -= 1
        if window_length < 1:
            raise ValueError("Resulting window_length is < 1; can't apply filter")

        filtered = np.array(
            savgol_filter(
                np.asarray(inten_list, dtype=float), window_length, poly_order
            )
        )

        return self.__class__.from_arrays(wl_list, filtered.tolist())

    def filter_by_wavelength(self, wl_min: float, wl_max: float) -> Self:
        if wl_min > wl_max:
            raise ValueError("wl_min must be <= wl_max")

        filtered = [(wl, inten) for wl, inten in self.data if wl_min <= wl <= wl_max]
        return self.__class__(filtered)
