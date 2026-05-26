"""
Fast transmission-matrix helper for pyMMF Modes.

This module is kept in the same folder as data_maker.ipynb so remote runs can
import it without hard-coded local paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.linalg import expm


Curvature = Optional[Sequence[Optional[float]]]


@dataclass
class FastTMBuilder:
    """Cached helper for curved pyMMF propagation matrices."""

    modes: object
    npola: int = 1
    dtype: np.dtype = np.complex128
    cache_y: bool = False

    def __post_init__(self) -> None:
        if self.npola != 1:
            raise NotImplementedError("FastTMBuilder currently supports npola=1 only.")

        if getattr(self.modes, "wl", None) is None:
            raise ValueError("modes.wl is missing.")
        if getattr(self.modes, "indexProfile", None) is None:
            raise ValueError("modes.indexProfile is missing.")

        self.M = self.modes.getModeMatrix(npola=self.npola).astype(self.dtype, copy=False)
        self.betas = np.asarray(self.modes.betas, dtype=np.float64)
        self.B0 = np.diag(self.betas).astype(self.dtype)
        self.k0 = 2.0 * np.pi / float(self.modes.wl)
        self.n_min = float(np.min(self.modes.indexProfile.n))

        x_flat = np.asarray(self.modes.indexProfile.X, dtype=np.float64).ravel()
        self.Gamma_x = self._coordinate_overlap(x_flat)

        self.Gamma_y = None
        if self.cache_y:
            y_flat = np.asarray(self.modes.indexProfile.Y, dtype=np.float64).ravel()
            self.Gamma_y = self._coordinate_overlap(y_flat)

    def _coordinate_overlap(self, coord_flat: np.ndarray) -> np.ndarray:
        weighted = coord_flat[:, None] * self.M
        return (self.M.conj().T @ weighted).astype(self.dtype, copy=False)

    @staticmethod
    def normalize_curvature(curvature: Curvature) -> Optional[tuple[Optional[float], Optional[float]]]:
        if curvature is None:
            return None
        if isinstance(curvature, (int, float)):
            if float(curvature) == 0.0:
                raise ValueError("curvature radius cannot be 0.")
            return (float(curvature), None)
        if hasattr(curvature, "__len__") and len(curvature) == 2:
            cx, cy = curvature
            if cx == 0 or cy == 0:
                raise ValueError("curvature radius cannot be 0.")
            return (None if cx is None else float(cx), None if cy is None else float(cy))
        raise ValueError("curvature must be None, a scalar radius, or [radius_x, radius_y].")

    def evolution_operator(self, curvature: Curvature = None) -> np.ndarray:
        curv = self.normalize_curvature(curvature)
        if curv is None:
            return self.B0.copy()

        radius_x, radius_y = curv
        B = self.B0.copy()
        coef = self.n_min * self.k0

        if radius_x is not None:
            B -= coef / radius_x * self.Gamma_x

        if radius_y is not None:
            if self.Gamma_y is None:
                y_flat = np.asarray(self.modes.indexProfile.Y, dtype=np.float64).ravel()
                self.Gamma_y = self._coordinate_overlap(y_flat)
            B -= coef / radius_y * self.Gamma_y

        return B

    def propagation_expm(self, distance: float, curvature: Curvature = None) -> np.ndarray:
        B = self.evolution_operator(curvature)
        return expm(1j * B * distance)

    def propagation_eigh(self, distance: float, curvature: Curvature = None) -> np.ndarray:
        B = self.evolution_operator(curvature)
        B = 0.5 * (B + B.conj().T)
        vals, vecs = np.linalg.eigh(B)
        phases = np.exp(1j * vals * distance)
        return (vecs * phases[None, :]) @ vecs.conj().T

    def propagation(self, distance: float, curvature: Curvature = None, method: str = "eigh") -> np.ndarray:
        if method == "eigh":
            return self.propagation_eigh(distance, curvature)
        if method == "expm":
            return self.propagation_expm(distance, curvature)
        raise ValueError(f"Unknown fast TM method: {method}")


def relative_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.linalg.norm(reference - candidate) / (np.linalg.norm(reference) + 1e-30))


def unitarity_error(T: np.ndarray) -> float:
    eye = np.eye(T.shape[0], dtype=T.dtype)
    return float(np.linalg.norm(T.conj().T @ T - eye) / np.linalg.norm(eye))

