import copy
import math
import sys
from logging import getLogger

import numpy as np
import scipy

from .config import Kohyama21ModelConfig

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


logger = getLogger()


class Kohyama21Model:
    def __init__(self, config: Kohyama21ModelConfig):
        self.config = copy.deepcopy(config)
        self.dtype = np.float64
        self.initialize()

    def initialize(self):
        self._initialize_arrays()
        self._initialize_constants()
        self._initialize_weight_records()

    def run_simulation(self) -> tuple[np.ndarray, np.ndarray]:
        ntime_out = self.config.nt // self.config.istep_out
        T1_out = np.zeros((ntime_out + 1, self.config.ny), dtype=self.dtype)
        T2_out = np.zeros((ntime_out + 1, self.config.ny), dtype=self.dtype)

        cnt = 0
        T1_out[cnt] = self.T_1
        T2_out[cnt] = self.T_2

        for itime in tqdm(range(self.config.nt)):
            self._forward_one_step()
            if (itime + 1) % self.config.istep_out == 0:
                cnt += 1
                T1_out[cnt] = self.T_1
                T2_out[cnt] = self.T_2

        return T1_out, T2_out

    #
    # Private methods
    #

    def _initialize_arrays(self):
        self.y = np.linspace(
            0, self.config.Ly, self.config.ny, endpoint=True, dtype=self.dtype
        )

        dy = np.unique(np.diff(self.y))
        assert len(dy) == 1  # constant grid spacing
        self.dy = dy[0]

        # y-dependence of short wave radiation
        self.f = np.cos(np.pi * self.y / self.config.Ly).astype(self.dtype)

        self.T_1 = np.zeros(self.config.ny, dtype=self.dtype)  # temperature of basin 1
        self.T_2 = np.zeros(self.config.ny, dtype=self.dtype)

        # \bar{\theta} - \bar{\theta}_A
        self.theta = np.zeros(self.config.ny, dtype=self.dtype)
        self.tau_y = np.zeros(self.config.ny, dtype=self.dtype)  # d\tau/dy

        # boundary current stream function of basin 1
        self.psi_1 = np.zeros(self.config.ny, dtype=self.dtype)
        self.psi_2 = np.zeros(self.config.ny, dtype=self.dtype)

    def _initialize_constants(self):
        c = self.config
        self.d_e = c.d_l * c.D_a / (c.d_l + c.D_a)

        self.c_1 = c.beta * c.R_1**2
        self.c_2 = c.beta * c.R_2**2
        self.ratio_1 = c.Lx_1 / c.Lx
        self.ratio_2 = c.Lx_2 / c.Lx
        self.upsilon_1 = c.C_pw * c.rho_w * c.H_1 * 0.5 / (c.lamd * c.delta_1 * c.Lx_1)
        self.upsilon_2 = c.C_pw * c.rho_w * c.H_2 * 0.5 / (c.lamd * c.delta_2 * c.Lx_2)

        self.gamma = 1.0 / (
            c.B
            + (self.ratio_1 + self.ratio_2) * c.lamd
            + c.C_pa * c.rho_a * c.nu_a * self.d_e
        )

        self.a_1 = c.lamd / (c.C_pw * c.rho_w * c.H_1)
        self.a_2 = c.lamd / (c.C_pw * c.rho_w * c.H_2)

        self.b_1 = c.R_1**2 / (c.rho_w * c.H_1)
        self.b_2 = c.R_2**2 / (c.rho_w * c.H_2)

        self.to_1 = c.Lx_1 / self.c_1
        self.to_2 = c.Lx_2 / self.c_2

    def _initialize_weight_records(self):
        c = self.config
        n_record_1 = math.ceil(self.to_1 / c.dt)
        n_record_2 = math.ceil(self.to_2 / c.dt)

        self.weight_record_1 = _get_weight(n_record_1, c.dt, self.to_1)
        self.weight_record_2 = _get_weight(n_record_2, c.dt, self.to_2)

        self.tau_y_record_1 = np.zeros((c.ny, n_record_1), dtype=self.dtype)
        self.tau_y_record_2 = np.zeros((c.ny, n_record_2), dtype=self.dtype)

    def _forward_one_step(self):
        # Ocean is updated and then atmosphere is updated.
        # This order is important to preserve causality.
        # For ocearn, temperature is first calculated and then psi is updated.
        # The calculation of temperature is based on psi, but that of psi is not based on temperature.
        # So this ordering also preserves the causality.
        self._calculate_future_temperature()
        self._update_psi()
        self._update_atmosphere()
        self._update_tau_y_records()

    def _calculate_future_temperature(self):
        self.T_1 = self._calculate_future_temperature_for_one_basin(basin=1)
        self.T_2 = self._calculate_future_temperature_for_one_basin(basin=2)

    def _update_psi(self):
        # Since weight is normalized to 1 by to_1, multiplication of to_1 is needed.
        # This is the summation over time axis.
        self.psi_1 = (
            self.to_1
            * self.b_1
            * np.sum(self.tau_y_record_1 * self.weight_record_1, axis=1)
        )
        self.psi_2 = (
            self.to_2
            * self.b_2
            * np.sum(self.tau_y_record_2 * self.weight_record_2, axis=1)
        )

    def _update_atmosphere(self):
        c = self.config

        noise_theta_a = np.random.randn(c.ny) * c.amp_noise
        noise_theta_1 = np.random.randn(c.ny) * c.amp_noise_theta_1
        noise_theta_2 = np.random.randn(c.ny) * c.amp_noise_theta_2

        rhs1 = c.F * self.f
        rhs2 = self.ratio_1 * c.lamd * (self.T_1 - noise_theta_1)
        rhs3 = self.ratio_2 * c.lamd * (self.T_2 - noise_theta_2)
        self.theta = self.gamma * (rhs1 + rhs2 + rhs3) + noise_theta_a
        # = \bar{\theta} - \bar{\theta}_A

        rhs_tauy_1 = c.beta * (self.y - 0.5 * c.Ly)
        rhs_tauy_2 = c.f_0 * self.theta / (c.S_a * c.d_l)
        self.tau_y = self.d_e * c.rho_a * c.nu_a * (rhs_tauy_1 + rhs_tauy_2)
        # = d\tau / dy

    def _update_tau_y_records(self):
        # Roll along time axis
        self.tau_y_record_1 = np.roll(self.tau_y_record_1, shift=1, axis=1)
        # Update the most future state
        self.tau_y_record_1[:, 0] = self.tau_y

        self.tau_y_record_2 = np.roll(self.tau_y_record_2, shift=1, axis=1)
        self.tau_y_record_2[:, 0] = self.tau_y

    def _make_triangular_matrices(self, basin: int) -> tuple[np.ndarray, np.ndarray]:
        #
        if basin == 1:
            T = self.T_1
            psi = self.psi_1
            upsilon = self.upsilon_1
            a = self.a_1
        elif basin == 2:
            T = self.T_2
            psi = self.psi_2
            upsilon = self.upsilon_2
            a = self.a_2
        else:
            raise ValueError(f"Unexpected basin index {basin=}")

        c = self.config.dt / self.dy**2

        # psi at cell interface
        mid_psi_sq = (0.5 * (psi[1:] + psi[:-1])) ** 2
        offdiag_coeff = -c * (upsilon * mid_psi_sq + self.config.eps)

        diag = np.zeros_like(T)
        diag[1:-1] = (mid_psi_sq[1:] + mid_psi_sq[:-1]) * upsilon + 2 * self.config.eps

        # apply boundary condition (dT/dy = 0)
        # flux outside the domain is zero, so cancelation occurs and a half coeff is used.
        diag[0] = mid_psi_sq[0] * upsilon + self.config.eps
        diag[-1] = mid_psi_sq[-1] * upsilon + self.config.eps

        diag_coeff = c * diag + a * self.config.dt

        A_for_scipy = _make_triangular_matrix_for_sipy_solve_banded(
            diag_coeff, offdiag_coeff, offdiag_coeff
        )
        A = _make_triangular_matrix(diag_coeff, offdiag_coeff, offdiag_coeff)

        return A_for_scipy, A

    def _calculate_future_temperature_for_one_basin(self, basin: int) -> np.ndarray:
        # Crank-Nicolson method using scipy.linalg.solve_banded

        if basin == 1:
            T = self.T_1
            a = self.a_1
        elif basin == 2:
            T = self.T_2
            a = self.a_2
        else:
            raise ValueError(f"Unexpected basin index {basin=}")

        A_for_scipy, A = self._make_triangular_matrices(basin)

        A_for_scipy = 0.5 * A_for_scipy  # multiply by 0.5 for Crank-Nicolson method
        A_for_scipy[1, :] = 1.0 + A_for_scipy[1, :]  # add 1 to diag components

        B = T - 0.5 * A @ T + a * self.theta * self.config.dt

        return scipy.linalg.solve_banded(l_and_u=(1, 1), ab=A_for_scipy, b=B)


def _get_weight(n: int, dt: float, tmax: float) -> np.ndarray:
    assert n >= 2
    assert (n - 1) * dt <= tmax <= n * dt

    weight = np.zeros((n), dtype=np.float64)

    weight[0 : n - 1] = dt / tmax
    weight[n - 1] = (tmax - (n - 1) * dt) / tmax

    return weight


def _make_triangular_matrix(
    diag_coeff: np.ndarray,
    upper_offdiag_coeff: np.ndarray,
    lower_offdiag_coeff: np.ndarray,
) -> np.ndarray:
    #
    assert diag_coeff.ndim == upper_offdiag_coeff.ndim == lower_offdiag_coeff.ndim == 1
    assert (
        len(diag_coeff) == 1 + len(upper_offdiag_coeff) == 1 + len(lower_offdiag_coeff)
    )

    n = len(diag_coeff)
    A = np.zeros((n, n), dtype=np.float64)

    np.fill_diagonal(A, val=diag_coeff)

    i = np.arange(n - 1).astype(int)
    A[i, i + 1] = upper_offdiag_coeff
    A[i + 1, i] = lower_offdiag_coeff

    return A


def _make_triangular_matrix_for_sipy_solve_banded(
    diag_coeff: np.ndarray,
    upper_offdiag_coeff: np.ndarray,
    lower_offdiag_coeff: np.ndarray,
) -> np.ndarray:
    #
    assert diag_coeff.ndim == upper_offdiag_coeff.ndim == lower_offdiag_coeff.ndim == 1
    assert (
        len(diag_coeff) == 1 + len(upper_offdiag_coeff) == 1 + len(lower_offdiag_coeff)
    )

    n = len(diag_coeff)
    ab = np.zeros((3, n), dtype=diag_coeff.dtype)

    ab[0, 1:] = upper_offdiag_coeff
    ab[1, :] = diag_coeff
    ab[2, :-1] = lower_offdiag_coeff

    return ab
