#!/usr/bin/env python3
"""Generate and analyze the v5 inertial roll-identification study.

V5 reconciles the published and stored objective normalizations, exports the
actual antialiased analysis trace, and makes diagnostic line/noise ratios
explicit while retaining the v4.1 scientific design.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.signal import firwin, resample_poly


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DATA = ROOT / "data"
CONFIG_PATH = HERE / "analysis_config.json"


@dataclass(frozen=True)
class Payload:
    name: str
    mass: float
    inertia: float


@dataclass
class SpectralData:
    angular: np.ndarray
    input_spectrum: np.ndarray
    output_spectrum: np.ndarray
    road_spectrum: np.ndarray | None
    input_noise_power: float
    output_noise_power: float
    input_signal_power: float
    output_signal_power: float
    frequencies_used: int
    noise_bins_used: int
    estimator_mode: str


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def frequency_grid(spec: list[float]) -> np.ndarray:
    start, stop, step = spec
    count = int(round((stop - start) / step)) + 1
    return np.round(np.linspace(start, stop, count), 10)


class Study:
    def __init__(self, config: dict):
        self.c = config
        self.fs = int(config["base_sampling_hz"])
        self.duration = float(config["duration_s"])
        self.analysis_start = float(config["analysis_start_s"])
        self.analysis_duration = float(config["analysis_duration_s"])
        self.analysis_stop = self.analysis_start + self.analysis_duration
        self.g = float(config["gravity_mps2"])
        self.hr = float(config["roll_axis_height_m"])
        self.k_phi = float(config["roll_stiffness_Nm_per_rad"])
        self.c_phi = float(config["roll_damping_Nms_per_rad"])
        self.broad = frequency_grid(config["broadband_frequencies_hz"])
        self.low = frequency_grid(config["lowband_frequencies_hz"])
        self.n = int(round(self.duration * self.fs))
        self.t = np.arange(self.n, dtype=float) / self.fs
        self.payloads = {
            key: Payload(key, float(value["sprung_mass_kg"]),
                         float(value["roll_inertia_kgm2"]))
            for key, value in config["payloads"].items()
        }
        geometry = config["payload_geometry"]
        vertical = float(geometry["vertical_height_above_roll_axis_m"])
        lateral = float(geometry["lateral_offset_m"])
        radius = float(geometry["effective_roll_radius_m"])
        if not math.isclose(vertical, self.hr, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Payload height must preserve the declared h_r")
        if not math.isclose(
                vertical ** 2 + lateral ** 2, radius ** 2,
                rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Payload coordinates do not match effective radius")
        base = self.payloads["P0"]
        for name in ("P75", "P150"):
            payload = self.payloads[name]
            expected = base.inertia + (payload.mass - base.mass) * radius ** 2
            if not math.isclose(
                    payload.inertia, expected, rel_tol=0.0, abs_tol=1e-10):
                raise ValueError(f"Payload inertia is inconsistent for {name}")
        self.cache: dict[tuple, dict] = {}

    def effective_stiffness(self, payload: Payload) -> float:
        return self.k_phi - payload.mass * self.g * self.hr

    def components(self, seed: int, frequencies: np.ndarray, stream: int,
                   target_rms: float) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed * 1009 + 17 + stream)
        phases = rng.uniform(0.0, 2.0 * np.pi, len(frequencies))
        amplitudes = 1.0 / np.sqrt(frequencies)
        amplitudes *= target_rms / math.sqrt(float(np.sum(amplitudes ** 2)) / 2.0)
        return amplitudes, phases

    def basis(self, frequencies: np.ndarray, phases: np.ndarray) -> np.ndarray:
        angles = (2.0 * np.pi * frequencies[:, None] * self.t[None, :]
                  + phases[:, None])
        return np.exp(1j * angles)

    @staticmethod
    def synthesize(amplitudes: np.ndarray, basis: np.ndarray,
                   transfer: np.ndarray | None = None) -> np.ndarray:
        coefficients = amplitudes if transfer is None else amplitudes * transfer
        return np.imag(coefficients @ basis)

    def calibration_bank(self, seed: int, stream: int = 0) -> dict[str, float]:
        rng = np.random.default_rng(seed * 1013 + stream * 7919 + 29)
        return {
            "accel_scale": float(rng.normal(
                0.0, float(self.c["accelerometer_scale_std"]))),
            "gyro_scale": float(rng.normal(
                0.0, float(self.c["gyro_scale_std"]))),
        }

    def white_noise(self, seed: int, stream: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed * 104729 + stream * 1009 + 53)
        return {
            "accel": rng.normal(
                0.0, float(self.c["accelerometer_noise_std_mps2"]), self.n),
            "gyro": rng.normal(
                0.0, float(self.c["gyro_noise_std_radps"]), self.n),
        }

    def simulate(self, seed: int, payload_name: str = "P0",
                 excitation: str = "broad", road: bool = False,
                 excitation_stream: int = 0,
                 rms_scale: float = 1.0) -> dict:
        key = (seed, payload_name, excitation, road,
               excitation_stream, round(rms_scale, 12))
        if key in self.cache:
            return self.cache[key]

        payload = self.payloads[payload_name]
        frequencies = self.broad if excitation == "broad" else self.low
        target_rms = (float(self.c["lateral_acceleration_rms_mps2"])
                      * float(rms_scale))
        amplitudes, phases = self.components(
            seed, frequencies, excitation_stream, target_rms)
        excitation_basis = self.basis(frequencies, phases)
        ay = self.synthesize(amplitudes, excitation_basis)

        angular = 2.0 * np.pi * frequencies
        stiffness = self.effective_stiffness(payload)
        denominator = (stiffness - payload.inertia * angular ** 2
                       + 1j * self.c_phi * angular)
        phi_transfer_ay = payload.mass * self.hr / denominator

        phi = self.synthesize(amplitudes, excitation_basis, phi_transfer_ay)
        roll_rate = self.synthesize(
            amplitudes, excitation_basis, 1j * angular * phi_transfer_ay)
        roll_accel = self.synthesize(
            amplitudes, excitation_basis, -(angular ** 2) * phi_transfer_ay)

        road_proxy = np.zeros(self.n)
        road_moment = np.zeros(self.n)
        if road:
            q_amp, q_phase = self.components(seed, self.broad, 41, 1.0)
            q_basis = self.basis(self.broad, q_phase)
            road_proxy = self.synthesize(q_amp, q_basis)
            q_angular = 2.0 * np.pi * self.broad
            tau = float(self.c["road_transfer_time_constant_s"])
            road_filter = 1.0 / (1.0 + 1j * q_angular * tau)
            filtered = self.synthesize(q_amp, q_basis, road_filter)
            target_moment_rms = (
                float(self.c["road_moment_fraction_of_lateral_rms"])
                * payload.mass * self.hr * target_rms)
            scale = target_moment_rms / np.std(filtered)
            road_moment = filtered * scale
            road_coefficients = q_amp * road_filter * scale
            road_denominator = (
                stiffness - payload.inertia * q_angular ** 2
                + 1j * self.c_phi * q_angular)
            phi_transfer_moment = 1.0 / road_denominator
            phi += self.synthesize(
                road_coefficients, q_basis, phi_transfer_moment)
            roll_rate += self.synthesize(
                road_coefficients, q_basis,
                1j * q_angular * phi_transfer_moment)
            roll_accel += self.synthesize(
                road_coefficients, q_basis,
                -(q_angular ** 2) * phi_transfer_moment)

        result = {
            "time": self.t,
            "ay_truth": ay,
            "phi": phi,
            "roll_rate_truth": roll_rate,
            "roll_accel": roll_accel,
            "road_proxy": road_proxy,
            "road_moment": road_moment,
            "frequencies": frequencies,
        }
        self.cache[key] = result
        return result

    def measurements(self, simulation: dict, seed: int, stream: int,
                     input_error: bool = True,
                     output_error: bool = True,
                     calibration_stream: int = 0) -> tuple[np.ndarray, np.ndarray]:
        calibration = self.calibration_bank(seed, calibration_stream)
        noise = self.white_noise(seed, stream)
        if input_error:
            ay = ((1.0 + calibration["accel_scale"])
                  * simulation["ay_truth"] + noise["accel"])
        else:
            ay = np.asarray(simulation["ay_truth"])
        if output_error:
            gyro = ((1.0 + calibration["gyro_scale"])
                    * simulation["roll_rate_truth"] + noise["gyro"])
        else:
            gyro = np.asarray(simulation["roll_rate_truth"])
        return ay, gyro

    def elevated_specific_force(self, simulation: dict, seed: int,
                                stream: int, height: float,
                                calibration_stream: int = 0) -> np.ndarray:
        calibration = self.calibration_bank(seed, calibration_stream)
        noise = self.white_noise(seed, stream)
        truth = (simulation["ay_truth"] - height * simulation["roll_accel"]
                 + self.g * np.sin(simulation["phi"]))
        return ((1.0 + calibration["accel_scale"]) * truth + noise["accel"])

    def measured_road_proxy(self, simulation: dict, seed: int) -> np.ndarray:
        """Return a deliberately imperfect road-proxy measurement."""
        proxy = np.asarray(simulation["road_proxy"])
        delay = float(self.c["road_proxy_delay_s"])
        delayed = np.interp(
            self.t - delay, self.t, proxy, left=proxy[0], right=proxy[-1])
        rng = np.random.default_rng(seed * 65537 + 607)
        noise_std = (float(self.c["road_proxy_noise_fraction_of_rms"])
                     * float(np.std(proxy)))
        return delayed + rng.normal(0.0, noise_std, self.n)

    def analysis_slice(self, sample_rate: int) -> slice:
        first = int(round(self.analysis_start * sample_rate))
        last = int(round(self.analysis_stop * sample_rate))
        return slice(first, last)


def polyphase_filter(factor: int, beta: float = 5.0) -> np.ndarray:
    """Reproduce SciPy's documented default low-pass design explicitly."""
    if factor < 1:
        raise ValueError("Decimation factor must be positive")
    return firwin(
        20 * factor + 1, 1.0 / factor, window=("kaiser", beta))


def decimate(signal: np.ndarray, base_rate: int, target_rate: int,
             antialias: bool = True) -> np.ndarray:
    factor = base_rate // target_rate
    if base_rate % target_rate:
        raise ValueError("All configured sample rates must divide the base rate")
    if antialias:
        taps = polyphase_filter(factor)
        return resample_poly(
            signal, 1, factor, window=taps, padtype="constant")
    return signal[::factor]


def active_bounds(values: np.ndarray, lower: np.ndarray,
                  upper: np.ndarray) -> str:
    names = ("I_xx", "C_phi", "K_fit", "Gamma_M")
    active = []
    for index, value in enumerate(values):
        span = upper[index] - lower[index]
        if min(value - lower[index], upper[index] - value) <= 1e-4 * span:
            active.append(names[index])
    return ";".join(active) if active else "none"


def prepare_spectra(ay: np.ndarray, gyro: np.ndarray, sample_rate: int,
                    frequencies: np.ndarray, config: dict,
                    road_proxy: np.ndarray | None = None,
                    estimator_mode: str = "estimated_eiv",
                    injected_input_error: np.ndarray | None = None,
                    injected_output_error: np.ndarray | None = None) -> SpectralData:
    count = len(ay)
    window = np.hanning(count)
    input_fft = np.fft.rfft((ay - np.mean(ay)) * window)
    output_fft = np.fft.rfft((gyro - np.mean(gyro)) * window)
    fft_frequencies = np.fft.rfftfreq(count, 1.0 / sample_rate)
    maximum = min(
        float(np.max(frequencies)),
        float(config["eiv_nyquist_fraction"]) * sample_rate / 2.0)
    selected = frequencies[frequencies <= maximum + 1e-10]
    if len(selected) < 4:
        raise RuntimeError("Too few excited frequencies for a three-parameter fit")
    indices = np.asarray([
        np.argmin(np.abs(fft_frequencies - value)) for value in selected
    ], dtype=int)

    candidate = np.where(
        (fft_frequencies > 0.0)
        & (fft_frequencies <= maximum + 1e-10))[0]
    guard = int(config["eiv_noise_guard_bins"])
    excluded: set[int] = set()
    for index in indices:
        excluded.update(range(max(0, index - guard), index + guard + 1))
    noise_indices = np.asarray([i for i in candidate if i not in excluded], dtype=int)
    if len(noise_indices) < 10:
        candidate = np.arange(1, len(fft_frequencies))
        noise_indices = np.asarray([i for i in candidate if i not in excluded], dtype=int)

    output_scale = max(
        float(np.sqrt(np.mean(np.abs(output_fft[indices]) ** 2))), 1e-12)
    normalized_input = input_fft[indices] / output_scale
    normalized_output = output_fft[indices] / output_scale
    if estimator_mode == "injected_noise_scalar_eiv":
        if injected_input_error is None or injected_output_error is None:
            raise ValueError(
                "Injected-noise scalar EIV requires both additive-error sequences")
        injected_input_fft = np.fft.rfft(
            (injected_input_error - np.mean(injected_input_error)) * window)
        injected_output_fft = np.fft.rfft(
            (injected_output_error - np.mean(injected_output_error)) * window)
        input_noise = float(np.mean(np.abs(injected_input_fft[candidate]) ** 2)
                            / output_scale ** 2)
        output_noise = float(np.mean(np.abs(injected_output_fft[candidate]) ** 2)
                             / output_scale ** 2)
    else:
        input_noise = float(np.median(np.abs(input_fft[noise_indices]) ** 2)
                            / output_scale ** 2)
        output_noise = float(np.median(np.abs(output_fft[noise_indices]) ** 2)
                             / output_scale ** 2)
    if estimator_mode == "output_error":
        input_noise = 0.0
    elif estimator_mode not in ("estimated_eiv", "injected_noise_scalar_eiv"):
        raise ValueError(f"Unknown estimator mode: {estimator_mode}")
    input_noise = max(input_noise, 1e-16)
    output_noise = max(output_noise, 1e-16)

    q_spectrum = None
    if road_proxy is not None:
        q_fft = np.fft.rfft((road_proxy - np.mean(road_proxy)) * window)
        q_spectrum = q_fft[indices] / output_scale

    return SpectralData(
        angular=2.0 * np.pi * fft_frequencies[indices],
        input_spectrum=normalized_input,
        output_spectrum=normalized_output,
        road_spectrum=q_spectrum,
        input_noise_power=input_noise,
        output_noise_power=output_noise,
        input_signal_power=float(np.mean(np.abs(normalized_input) ** 2)),
        output_signal_power=float(np.mean(np.abs(normalized_output) ** 2)),
        frequencies_used=len(selected),
        noise_bins_used=len(noise_indices),
        estimator_mode=estimator_mode,
    )


def transfer_components(parameters: np.ndarray, data: SpectralData,
                        mass: float, hr: float) -> tuple[np.ndarray, np.ndarray]:
    inertia, damping, stiffness = parameters[:3]
    denominator = (stiffness - inertia * data.angular ** 2
                   + 1j * damping * data.angular)
    h_input = 1j * data.angular * mass * hr / denominator
    h_moment = 1j * data.angular / denominator
    return h_input, h_moment


def eiv_residual(parameters: np.ndarray, data: SpectralData,
                 mass: float, hr: float) -> np.ndarray:
    h_input, h_moment = transfer_components(parameters, data, mass, hr)
    prediction = h_input * data.input_spectrum
    if data.road_spectrum is not None:
        prediction = prediction + h_moment * parameters[3] * data.road_spectrum
    variance = (data.output_noise_power
                + np.abs(h_input) ** 2 * data.input_noise_power)
    standardized = (prediction - data.output_spectrum) / np.sqrt(variance)
    return np.concatenate((standardized.real, standardized.imag))


def fit_spectra(data: SpectralData, mass: float, hr: float,
                starts_override: tuple | None = None) -> dict:
    if data.road_spectrum is None:
        lower = np.asarray([200.0, 500.0, 20000.0])
        upper = np.asarray([1200.0, 10000.0, 100000.0])
        starts = starts_override or (
            [450.0, 2500.0, 45000.0],
            [700.0, 5000.0, 65000.0],
            [1000.0, 8000.0, 85000.0],
        )
    else:
        lower = np.asarray([200.0, 500.0, 20000.0, -200000.0])
        upper = np.asarray([1200.0, 10000.0, 100000.0, 200000.0])
        starts = starts_override or (
            [450.0, 2500.0, 45000.0, 0.0],
            [700.0, 5000.0, 65000.0, 30000.0],
            [1000.0, 8000.0, 85000.0, -30000.0],
        )

    solutions = []
    for start in starts:
        solution = least_squares(
            eiv_residual, start, args=(data, mass, hr), bounds=(lower, upper),
            x_scale="jac", max_nfev=800, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        solutions.append(solution)
    best = min(solutions, key=lambda item: float(np.sum(item.fun ** 2)))

    parameter_scale = np.maximum(np.abs(best.x), 1.0)
    normalized_jacobian = best.jac * parameter_scale[np.newaxis, :]
    singular_values = np.linalg.svd(normalized_jacobian, compute_uv=False)
    condition = (float(singular_values[0] / singular_values[-1])
                 if singular_values[-1] > 0.0 else math.inf)
    covariance_shape = np.linalg.pinv(normalized_jacobian.T @ normalized_jacobian)
    diagonal = np.sqrt(np.maximum(np.diag(covariance_shape), 1e-30))
    correlation = covariance_shape / np.outer(diagonal, diagonal)
    correlation_no_diagonal = np.abs(correlation - np.eye(len(best.x)))
    maximum_index = np.unravel_index(
        np.argmax(correlation_no_diagonal), correlation_no_diagonal.shape)
    names = ("I_xx", "C_phi", "K_fit", "Gamma_M")

    h_input, h_moment = transfer_components(best.x, data, mass, hr)
    prediction = h_input * data.input_spectrum
    if data.road_spectrum is not None:
        prediction += h_moment * best.x[3] * data.road_spectrum
    relative_residual = float(
        np.sqrt(np.mean(np.abs(prediction - data.output_spectrum) ** 2))
        / max(np.sqrt(np.mean(np.abs(data.output_spectrum) ** 2)), 1e-12))

    complex_objective = float(2.0 * np.mean(best.fun ** 2))
    input_line_noise_ratio = (
        math.nan if data.estimator_mode == "output_error" else
        float(10.0 * np.log10(
            data.input_signal_power / data.input_noise_power)))
    output_line_noise_ratio = float(10.0 * np.log10(
        data.output_signal_power / data.output_noise_power))

    return {
        "parameters": best.x,
        "eiv_objective": complex_objective,
        "spectral_residual": relative_residual,
        "normalized_condition": condition,
        "max_abs_correlation": float(correlation_no_diagonal[maximum_index]),
        "max_correlation_pair": f"{names[maximum_index[0]]}:{names[maximum_index[1]]}",
        "frequencies_used": data.frequencies_used,
        "noise_bins_used": data.noise_bins_used,
        "estimator_mode": data.estimator_mode,
        "input_noise_power": data.input_noise_power,
        "output_noise_power": data.output_noise_power,
        "input_line_noise_ratio_db": input_line_noise_ratio,
        "output_line_noise_ratio_db": output_line_noise_ratio,
        "active_bounds": active_bounds(best.x, lower, upper),
    }


def fit_frequency_domain(ay: np.ndarray, gyro: np.ndarray, sample_rate: int,
                         mass: float, hr: float, frequencies: np.ndarray,
                         config: dict, road_proxy: np.ndarray | None = None,
                         estimator_mode: str = "estimated_eiv",
                         injected_input_error: np.ndarray | None = None,
                         injected_output_error: np.ndarray | None = None) -> dict:
    data = prepare_spectra(
        ay, gyro, sample_rate, frequencies, config, road_proxy,
        estimator_mode, injected_input_error, injected_output_error)
    return fit_spectra(data, mass, hr)


def profile_inertia(data: SpectralData, mass: float, hr: float,
                    best_parameters: np.ndarray, best_objective: float,
                    inertia_grid: np.ndarray) -> list[dict]:
    rows = []
    lower = np.asarray([500.0, 20000.0])
    upper = np.asarray([10000.0, 100000.0])
    start = np.clip(best_parameters[1:3], lower, upper)

    for inertia in inertia_grid:
        def fixed_residual(other: np.ndarray) -> np.ndarray:
            return eiv_residual(
                np.asarray([inertia, other[0], other[1]]), data, mass, hr)

        solution = least_squares(
            fixed_residual, start, bounds=(lower, upper), x_scale="jac",
            max_nfev=500, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        start = solution.x
        rows.append({
            "I_xx_fixed_kgm2": float(inertia),
            "C_phi_profile_Nms_rad": float(solution.x[0]),
            "K_fit_profile_Nm_rad": float(solution.x[1]),
            "profile_objective": float(2.0 * np.mean(solution.fun ** 2)),
        })
    minimum = min(row["profile_objective"] for row in rows)
    if minimum < best_objective * (1.0 - 1e-7):
        raise RuntimeError("Fixed-inertia profile improved the full fit")
    for row in rows:
        row["relative_profile_objective"] = (
            row["profile_objective"] / best_objective)
    return rows


def profile_region(rows: list[dict], threshold: float,
                   lower_bound: float, upper_bound: float) -> dict:
    """Return the threshold component containing the profile minimum.

    Bounds touching the component are reported explicitly as censoring rather
    than being mistaken for observed threshold crossings.
    """
    ordered = sorted(rows, key=lambda item: item["I_xx_fixed_kgm2"])
    x = np.asarray([row["I_xx_fixed_kgm2"] for row in ordered], dtype=float)
    y = np.asarray([row["relative_profile_objective"] for row in ordered],
                   dtype=float)
    center = int(np.argmin(y))
    left = center
    right = center
    while left > 0 and y[left - 1] <= threshold:
        left -= 1
    while right < len(y) - 1 and y[right + 1] <= threshold:
        right += 1

    left_censored = left == 0 and y[left] <= threshold
    right_censored = right == len(y) - 1 and y[right] <= threshold

    def crossing(x0: float, y0: float, x1: float, y1: float) -> float:
        if math.isclose(y0, y1):
            return 0.5 * (x0 + x1)
        return x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)

    lower = lower_bound if left_censored else crossing(
        x[left - 1], y[left - 1], x[left], y[left])
    upper = upper_bound if right_censored else crossing(
        x[right], y[right], x[right + 1], y[right + 1])
    return {
        "profile_lower_kgm2": float(lower),
        "profile_upper_kgm2": float(upper),
        "profile_span_within_bounds_kgm2": float(upper - lower),
        "left_censored": bool(left_censored),
        "right_censored": bool(right_censored),
        "censored": bool(left_censored or right_censored),
    }


def scored_row(model: Study, fit: dict, truth_payload: Payload,
               fit_id: str, **labels) -> dict:
    estimate = fit["parameters"]
    truth = np.asarray([
        truth_payload.inertia, model.c_phi,
        model.effective_stiffness(truth_payload)])
    errors = 100.0 * (estimate[:3] / truth - 1.0)
    row = {"fit_id": fit_id, **labels}
    row.update({
        "I_xx_est_kgm2": float(estimate[0]),
        "C_phi_est_Nms_rad": float(estimate[1]),
        "K_fit_est_Nm_rad": float(estimate[2]),
        "I_xx_error_pct": float(errors[0]),
        "C_phi_error_pct": float(errors[1]),
        "K_eff_error_pct": float(errors[2]),
        "abs_I_error": abs(float(errors[0])),
        "abs_C_error": abs(float(errors[1])),
        "abs_K_error": abs(float(errors[2])),
        "eiv_objective": fit["eiv_objective"],
        "spectral_residual": fit["spectral_residual"],
        "normalized_condition": fit["normalized_condition"],
        "max_abs_correlation": fit["max_abs_correlation"],
        "max_correlation_pair": fit["max_correlation_pair"],
        "frequencies_used": fit["frequencies_used"],
        "noise_bins_used": fit["noise_bins_used"],
        "estimator_mode": fit["estimator_mode"],
        "input_noise_power": fit["input_noise_power"],
        "output_noise_power": fit["output_noise_power"],
        "input_line_noise_ratio_db": fit["input_line_noise_ratio_db"],
        "output_line_noise_ratio_db": fit["output_line_noise_ratio_db"],
        "active_bounds": fit["active_bounds"],
    })
    if len(estimate) == 4:
        row["road_moment_gain_est_Nm"] = float(estimate[3])
    return row


def summarize(rows: list[dict], keys: list[str], group_name: str) -> dict:
    result = {"group": group_name, "runs": len(rows)}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows])
        finite = values[np.isfinite(values)]
        if finite.size:
            result[f"median_{key}"] = float(np.median(finite))
            result[f"p90_{key}"] = float(np.percentile(finite, 90.0))
            result[f"q025_{key}"] = float(np.percentile(finite, 2.5))
            result[f"q975_{key}"] = float(np.percentile(finite, 97.5))
        else:
            result[f"median_{key}"] = math.nan
            result[f"p90_{key}"] = math.nan
            result[f"q025_{key}"] = math.nan
            result[f"q975_{key}"] = math.nan
    result["bound_active_fraction"] = float(np.mean(
        [row["active_bounds"] != "none" for row in rows]))
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run() -> None:
    config = load_config()
    study = Study(config)
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    seeds = range(1, int(config["monte_carlo_seeds"]) + 1)
    base_rate = int(config["base_sampling_hz"])
    p0 = study.payloads["P0"]
    all_rows: list[dict] = []

    # Rate study: one combined-noise high-rate record per seed.
    rate_rows: list[dict] = []
    baseline_by_seed: dict[int, dict] = {}
    for seed in seeds:
        simulation = study.simulate(seed, "P0", "broad", road=False)
        ay_full, gyro_full = study.measurements(simulation, seed, 0, True, True)
        for rate_value in config["sampling_rates_hz"]:
            rate = int(rate_value)
            interval = study.analysis_slice(rate)
            ay = decimate(ay_full, base_rate, rate, True)[interval]
            gyro = decimate(gyro_full, base_rate, rate, True)[interval]
            fit = fit_frequency_domain(
                ay, gyro, rate, p0.mass, study.hr, study.broad, config)
            row = scored_row(
                study, fit, p0, f"rate_s{seed:03d}_{rate}Hz_aa",
                study="rate", seed=seed, payload="P0", sample_rate_hz=rate,
                excitation="broad", input_channel="roll_axis_reference",
                preprocessing="antialiased")
            rate_rows.append(row)
            all_rows.append(row)
            if rate == 100:
                baseline_by_seed[seed] = row

        rate = 2
        interval = study.analysis_slice(rate)
        ay = decimate(ay_full, base_rate, rate, False)[interval]
        gyro = decimate(gyro_full, base_rate, rate, False)[interval]
        fit = fit_frequency_domain(
            ay, gyro, rate, p0.mass, study.hr, study.broad, config)
        row = scored_row(
            study, fit, p0, f"rate_s{seed:03d}_2Hz_naive",
            study="rate", seed=seed, payload="P0", sample_rate_hz=2,
            excitation="broad", input_channel="roll_axis_reference",
            preprocessing="naive")
        rate_rows.append(row)
        all_rows.append(row)

    # Noise decomposition and noise-free regression controls at 100 Hz.
    noise_rows: list[dict] = []
    for seed in seeds:
        simulation = study.simulate(seed, "P0", "broad", road=False)
        for label, input_error, output_error in (
                ("noise_free", False, False),
                ("input_only", True, False),
                ("output_only", False, True)):
            ay_full, gyro_full = study.measurements(
                simulation, seed, 0, input_error, output_error)
            interval = study.analysis_slice(100)
            ay = decimate(ay_full, base_rate, 100, True)[interval]
            gyro = decimate(gyro_full, base_rate, 100, True)[interval]
            fit = fit_frequency_domain(
                ay, gyro, 100, p0.mass, study.hr, study.broad, config)
            row = scored_row(
                study, fit, p0, f"noise_s{seed:03d}_{label}",
                study="noise_control", seed=seed, payload="P0",
                sample_rate_hz=100, excitation="broad",
                input_channel="roll_axis_reference", noise_condition=label)
            noise_rows.append(row)
            all_rows.append(row)

    # Estimator benchmark: output error and injected-noise scalar-power EIV
    # are compared with the estimated-noise EIV baseline on identical records.
    estimator_rows = [
        {**baseline_by_seed[seed], "estimator_view": "estimated_eiv"}
        for seed in seeds]
    for seed in seeds:
        simulation = study.simulate(seed, "P0", "broad", road=False)
        ay_full, gyro_full = study.measurements(
            simulation, seed, 0, True, True)
        additive = study.white_noise(seed, 0)
        interval = study.analysis_slice(100)
        ay = decimate(ay_full, base_rate, 100, True)[interval]
        gyro = decimate(gyro_full, base_rate, 100, True)[interval]
        input_error = decimate(
            additive["accel"], base_rate, 100, True)[interval]
        output_error = decimate(
            additive["gyro"], base_rate, 100, True)[interval]
        for mode in ("output_error", "injected_noise_scalar_eiv"):
            fit = fit_frequency_domain(
                ay, gyro, 100, p0.mass, study.hr, study.broad, config,
                estimator_mode=mode,
                injected_input_error=(
                    input_error if mode == "injected_noise_scalar_eiv" else None),
                injected_output_error=(
                    output_error if mode == "injected_noise_scalar_eiv" else None))
            row = scored_row(
                study, fit, p0, f"estimator_s{seed:03d}_{mode}",
                study="estimator_control", seed=seed, payload="P0",
                sample_rate_hz=100, excitation="broad",
                input_channel="roll_axis_reference",
                estimator_view=mode)
            estimator_rows.append(row)
            all_rows.append(row)

    # Placement study: matched motion/noise across four candidate heights.
    placement_rows: list[dict] = []
    for seed in seeds:
        simulation = study.simulate(seed, "P0", "broad", road=False)
        _, gyro_full = study.measurements(simulation, seed, 0, True, True)
        interval = study.analysis_slice(100)
        gyro = decimate(gyro_full, base_rate, 100, True)[interval]
        for height_value in config["sensor_heights_m"]:
            height = float(height_value)
            specific_force = study.elevated_specific_force(
                simulation, seed, 0, height)
            ay = decimate(specific_force, base_rate, 100, True)[interval]
            fit = fit_frequency_domain(
                ay, gyro, 100, p0.mass, study.hr, study.broad, config)
            row = scored_row(
                study, fit, p0, f"placement_s{seed:03d}_h{height:.1f}",
                study="placement", seed=seed, payload="P0",
                sample_rate_hz=100, excitation="broad",
                input_channel="uncompensated_specific_force",
                sensor_height_m=height)
            predicted_i = p0.inertia - p0.mass * study.hr * height
            predicted_k = study.k_phi
            row["predicted_apparent_I_xx_kgm2"] = predicted_i
            row["predicted_apparent_K_Nm_rad"] = predicted_k
            row["I_error_vs_apparent_pct"] = 100.0 * (
                row["I_xx_est_kgm2"] / predicted_i - 1.0)
            row["K_error_vs_apparent_pct"] = 100.0 * (
                row["K_fit_est_Nm_rad"] / predicted_k - 1.0)
            row["abs_I_error_vs_apparent"] = abs(row["I_error_vs_apparent_pct"])
            row["abs_K_error_vs_apparent"] = abs(row["K_error_vs_apparent_pct"])
            placement_rows.append(row)
            all_rows.append(row)

    # Excitation study: broadband baseline is referenced, not duplicated.
    excitation_rows = [
        {**baseline_by_seed[seed], "study_view": "excitation_broad"}
        for seed in seeds]
    for seed in seeds:
        simulation = study.simulate(seed, "P0", "low", road=False)
        ay_full, gyro_full = study.measurements(simulation, seed, 0, True, True)
        interval = study.analysis_slice(100)
        ay = decimate(ay_full, base_rate, 100, True)[interval]
        gyro = decimate(gyro_full, base_rate, 100, True)[interval]
        fit = fit_frequency_domain(
            ay, gyro, 100, p0.mass, study.hr, study.low, config)
        row = scored_row(
            study, fit, p0, f"excitation_s{seed:03d}_low",
            study="excitation", seed=seed, payload="P0", sample_rate_hz=100,
            excitation="low", input_channel="roll_axis_reference")
        excitation_rows.append(row)
        all_rows.append(row)

    # Payload study: three explicitly different pairing assumptions.
    payload_rows: list[dict] = []
    payload_estimates: dict[tuple[int, str, str], float] = {}
    for seed in seeds:
        base = {**baseline_by_seed[seed], "pairing": "independent_white"}
        payload_rows.append(base)
        payload_estimates[(seed, "independent_white", "P0")] = base["I_xx_est_kgm2"]
        payload_estimates[(seed, "common_acquisition", "P0")] = base["I_xx_est_kgm2"]
        payload_rows.append({**base, "pairing": "common_acquisition"})

        for payload_name, stream in (("P75", 75), ("P150", 150)):
            payload = study.payloads[payload_name]
            simulation = study.simulate(seed, payload_name, "broad", road=False)
            interval = study.analysis_slice(100)
            for pairing, noise_stream in (
                    ("independent_white", stream), ("common_acquisition", 0)):
                ay_full, gyro_full = study.measurements(
                    simulation, seed, noise_stream, True, True)
                ay = decimate(ay_full, base_rate, 100, True)[interval]
                gyro = decimate(gyro_full, base_rate, 100, True)[interval]
                fit = fit_frequency_domain(
                    ay, gyro, 100, payload.mass, study.hr, study.broad, config)
                row = scored_row(
                    study, fit, payload,
                    f"payload_s{seed:03d}_{payload_name}_{pairing}",
                    study="payload", seed=seed, payload=payload_name,
                    sample_rate_hz=100, excitation="broad",
                    input_channel="roll_axis_reference", pairing=pairing)
                payload_rows.append(row)
                all_rows.append(row)
                payload_estimates[(seed, pairing, payload_name)] = row["I_xx_est_kgm2"]

        # Independent acquisition changes multisine phases, RMS tracking,
        # persistent calibration, and white noise for every payload record.
        for payload_index, payload_name in enumerate(("P0", "P75", "P150")):
            payload = study.payloads[payload_name]
            stream = 300 + payload_index
            rng = np.random.default_rng(seed * 131071 + stream * 8191 + 73)
            rms_scale = 1.0 + rng.normal(
                0.0, float(config["payload_reacquisition_rms_std_fraction"]))
            simulation = study.simulate(
                seed, payload_name, "broad", road=False,
                excitation_stream=stream, rms_scale=rms_scale)
            interval = study.analysis_slice(100)
            ay_full, gyro_full = study.measurements(
                simulation, seed, stream, True, True,
                calibration_stream=stream)
            ay = decimate(ay_full, base_rate, 100, True)[interval]
            gyro = decimate(gyro_full, base_rate, 100, True)[interval]
            fit = fit_frequency_domain(
                ay, gyro, 100, payload.mass, study.hr, study.broad, config)
            row = scored_row(
                study, fit, payload,
                f"payload_s{seed:03d}_{payload_name}_independent_acquisition",
                study="payload", seed=seed, payload=payload_name,
                sample_rate_hz=100, excitation="broad",
                input_channel="roll_axis_reference",
                pairing="independent_acquisition",
                input_rms_scale=rms_scale)
            payload_rows.append(row)
            all_rows.append(row)
            payload_estimates[(seed, "independent_acquisition", payload_name)] = (
                row["I_xx_est_kgm2"])

    payload_differences: list[dict] = []
    for seed in seeds:
        for pairing in (
                "independent_white", "common_acquisition",
                "independent_acquisition"):
            base_estimate = payload_estimates[(seed, pairing, "P0")]
            for payload_name in ("P75", "P150"):
                estimate = (payload_estimates[(seed, pairing, payload_name)]
                            - base_estimate)
                truth = study.payloads[payload_name].inertia - p0.inertia
                payload_differences.append({
                    "seed": seed,
                    "pairing": pairing,
                    "comparison": f"{payload_name}-P0",
                    "delta_I_truth_kgm2": truth,
                    "delta_I_est_kgm2": estimate,
                    "delta_I_error_pct": 100.0 * (estimate / truth - 1.0),
                })

    # Dynamic road mismatch; zero-road baseline is referenced, not duplicated.
    road_rows = [
        {**baseline_by_seed[seed], "road_condition": "zero_road"}
        for seed in seeds]
    for seed in seeds:
        simulation = study.simulate(seed, "P0", "broad", road=True)
        ay_full, gyro_full = study.measurements(simulation, seed, 0, True, True)
        interval = study.analysis_slice(100)
        ay = decimate(ay_full, base_rate, 100, True)[interval]
        gyro = decimate(gyro_full, base_rate, 100, True)[interval]
        proxy = decimate(simulation["road_proxy"], base_rate, 100, True)[interval]
        measured_proxy_full = study.measured_road_proxy(simulation, seed)
        measured_proxy = decimate(
            measured_proxy_full, base_rate, 100, True)[interval]
        fit = fit_frequency_domain(
            ay, gyro, 100, p0.mass, study.hr, study.broad, config)
        row = scored_row(
            study, fit, p0, f"road_s{seed:03d}_unmeasured",
            study="road", seed=seed, payload="P0", sample_rate_hz=100,
            excitation="broad", input_channel="roll_axis_reference",
            road_condition="dynamic_road_unmeasured")
        road_rows.append(row)
        all_rows.append(row)

        fit_proxy = fit_frequency_domain(
            ay, gyro, 100, p0.mass, study.hr, study.broad, config,
            road_proxy=proxy)
        row = scored_row(
            study, fit_proxy, p0, f"road_s{seed:03d}_static_proxy",
            study="road", seed=seed, payload="P0", sample_rate_hz=100,
            excitation="broad", input_channel="roll_axis_reference",
            road_condition="dynamic_road_exact_static_proxy")
        road_rows.append(row)
        all_rows.append(row)

        fit_measured_proxy = fit_frequency_domain(
            ay, gyro, 100, p0.mass, study.hr, study.broad, config,
            road_proxy=measured_proxy)
        row = scored_row(
            study, fit_measured_proxy, p0,
            f"road_s{seed:03d}_noisy_delayed_static_proxy",
            study="road", seed=seed, payload="P0", sample_rate_hz=100,
            excitation="broad", input_channel="roll_axis_reference",
            road_condition="dynamic_road_noisy_delayed_static_proxy")
        road_rows.append(row)
        all_rows.append(row)

    # Assert that the headline count contains no repeated fit identifiers.
    fit_ids = [row["fit_id"] for row in all_rows]
    if len(fit_ids) != len(set(fit_ids)):
        raise RuntimeError("Duplicate fit identifiers remain in all_fits.csv")

    write_csv(RESULTS / "all_fits.csv", all_rows)
    write_csv(RESULTS / "rate_study.csv", rate_rows)
    write_csv(RESULTS / "noise_controls.csv", noise_rows)
    write_csv(RESULTS / "estimator_controls.csv", estimator_rows)
    write_csv(RESULTS / "placement_study.csv", placement_rows)
    write_csv(RESULTS / "excitation_study.csv", excitation_rows)
    write_csv(RESULTS / "payload_study.csv", payload_rows)
    write_csv(RESULTS / "payload_differences.csv", payload_differences)
    write_csv(RESULTS / "road_mismatch_study.csv", road_rows)

    road_paired_rows: list[dict] = []
    for seed in seeds:
        conditions = {
            row["road_condition"]: row for row in road_rows
            if int(row["seed"]) == seed
        }
        for comparison, first, second in (
                ("exact_proxy_minus_unmeasured",
                 "dynamic_road_unmeasured",
                 "dynamic_road_exact_static_proxy"),
                ("perturbed_proxy_minus_exact_proxy",
                 "dynamic_road_exact_static_proxy",
                 "dynamic_road_noisy_delayed_static_proxy")):
            a = conditions[first]
            b = conditions[second]
            road_paired_rows.append({
                "seed": seed,
                "comparison": comparison,
                "delta_abs_I_error_pct_points": (
                    float(b["abs_I_error"]) - float(a["abs_I_error"])),
                "delta_abs_C_error_pct_points": (
                    float(b["abs_C_error"]) - float(a["abs_C_error"])),
                "delta_abs_K_error_pct_points": (
                    float(b["abs_K_error"]) - float(a["abs_K_error"])),
                "delta_normalized_condition": (
                    float(b["normalized_condition"])
                    - float(a["normalized_condition"])),
            })
    write_csv(RESULTS / "road_paired_differences.csv", road_paired_rows)
    road_paired_summary: list[dict] = []
    for comparison in (
            "exact_proxy_minus_unmeasured",
            "perturbed_proxy_minus_exact_proxy"):
        group = [row for row in road_paired_rows
                 if row["comparison"] == comparison]
        summary = {"comparison": comparison, "runs": len(group)}
        for key in (
                "delta_abs_I_error_pct_points",
                "delta_abs_C_error_pct_points",
                "delta_abs_K_error_pct_points",
                "delta_normalized_condition"):
            values = np.asarray([float(row[key]) for row in group])
            summary[f"median_{key}"] = float(np.median(values))
            summary[f"q025_{key}"] = float(np.percentile(values, 2.5))
            summary[f"q975_{key}"] = float(np.percentile(values, 97.5))
        road_paired_summary.append(summary)
    write_csv(RESULTS / "road_paired_summary.csv", road_paired_summary)

    keys = ["abs_I_error", "abs_C_error", "abs_K_error",
            "normalized_condition", "max_abs_correlation",
            "spectral_residual", "eiv_objective",
            "input_line_noise_ratio_db", "output_line_noise_ratio_db"]
    summaries: list[dict] = []
    for rate_value in config["sampling_rates_hz"]:
        rate = int(rate_value)
        group = [row for row in rate_rows
                 if int(row["sample_rate_hz"]) == rate
                 and row["preprocessing"] == "antialiased"]
        summaries.append(summarize(group, keys, f"rate_{rate}Hz_antialiased"))
    summaries.append(summarize(
        [row for row in rate_rows if row["preprocessing"] == "naive"],
        keys, "rate_2Hz_naive"))
    for height_value in config["sensor_heights_m"]:
        height = float(height_value)
        group = [row for row in placement_rows
                 if math.isclose(float(row["sensor_height_m"]), height)]
        summaries.append(summarize(
            group, keys + ["abs_I_error_vs_apparent",
                           "abs_K_error_vs_apparent"],
            f"placement_{height}m"))
    summaries.append(summarize(
        list(baseline_by_seed.values()), keys, "excitation_broad"))
    summaries.append(summarize(
        [row for row in excitation_rows if row.get("excitation") == "low"],
        keys, "excitation_low"))
    summaries.append(summarize(
        list(baseline_by_seed.values()), keys, "road_zero_road"))
    for condition in (
            "dynamic_road_unmeasured", "dynamic_road_exact_static_proxy",
            "dynamic_road_noisy_delayed_static_proxy"):
        summaries.append(summarize(
            [row for row in road_rows if row["road_condition"] == condition],
            keys, f"road_{condition}"))
    summaries.append(summarize(
        list(baseline_by_seed.values()), keys, "noise_combined"))
    for condition in ("noise_free", "input_only", "output_only"):
        summaries.append(summarize(
            [row for row in noise_rows if row["noise_condition"] == condition],
            keys, f"noise_{condition}"))
    for mode in (
            "estimated_eiv", "output_error", "injected_noise_scalar_eiv"):
        summaries.append(summarize(
            [row for row in estimator_rows if row["estimator_view"] == mode],
            keys, f"estimator_{mode}"))
    write_csv(RESULTS / "group_summary.csv", summaries)

    difference_summary: list[dict] = []
    for pairing in (
            "independent_white", "common_acquisition",
            "independent_acquisition"):
        for comparison in ("P75-P0", "P150-P0"):
            group = [row for row in payload_differences
                     if row["pairing"] == pairing
                     and row["comparison"] == comparison]
            signed = np.asarray([float(row["delta_I_error_pct"]) for row in group])
            absolute = np.abs(signed)
            difference_summary.append({
                "pairing": pairing,
                "comparison": comparison,
                "runs": len(group),
                "median_abs_delta_I_error_pct": float(np.median(absolute)),
                "p90_abs_delta_I_error_pct": float(np.percentile(absolute, 90.0)),
                "q025_signed_delta_I_error_pct": float(np.percentile(signed, 2.5)),
                "q975_signed_delta_I_error_pct": float(np.percentile(signed, 97.5)),
            })
    write_csv(RESULTS / "payload_difference_summary.csv", difference_summary)

    # Profile objectives for every seed. The 5% rule is a visualization
    # convention, not a confidence or likelihood threshold. Threshold regions
    # that reach the declared parameter bounds are explicitly censored.
    profile_rows: list[dict] = []
    profile_widths: list[dict] = []
    threshold = float(config["profile_relative_objective_threshold"])
    profile_points = int(config["profile_grid_points"])
    profile_lower = 200.0
    profile_upper = 1200.0
    base_inertia_grid = np.linspace(profile_lower, profile_upper, profile_points)
    for seed in seeds:
        for excitation, frequencies in (
                ("broad", study.broad), ("low", study.low)):
            simulation = study.simulate(seed, "P0", excitation, road=False)
            ay_full, gyro_full = study.measurements(
                simulation, seed, 0, True, True)
            interval = study.analysis_slice(100)
            ay = decimate(ay_full, base_rate, 100, True)[interval]
            gyro = decimate(gyro_full, base_rate, 100, True)[interval]
            data = prepare_spectra(ay, gyro, 100, frequencies, config)
            best = fit_spectra(data, p0.mass, study.hr)
            inertia_grid = base_inertia_grid.copy()
            nearest = int(np.argmin(np.abs(
                inertia_grid - float(best["parameters"][0]))))
            inertia_grid[nearest] = float(best["parameters"][0])
            inertia_grid = np.sort(inertia_grid)
            rows = profile_inertia(
                data, p0.mass, study.hr, best["parameters"],
                best["eiv_objective"], inertia_grid)
            optimum = min(rows, key=lambda item: item["profile_objective"])
            region = profile_region(
                rows, threshold, profile_lower, profile_upper)
            profile_widths.append({
                "seed": seed,
                "excitation": excitation,
                "relative_objective_threshold": threshold,
                **region,
                "profile_optimum_kgm2": optimum["I_xx_fixed_kgm2"],
                "full_fit_optimum_kgm2": float(best["parameters"][0]),
                "full_fit_objective": float(best["eiv_objective"]),
            })
            for row in rows:
                row["seed"] = seed
                row["excitation"] = excitation
                profile_rows.append(row)
    write_csv(RESULTS / "inertia_profiles.csv", profile_rows)
    write_csv(RESULTS / "inertia_profile_widths.csv", profile_widths)

    selected = study.simulate(1, "P0", "broad", road=False)
    selected_ay, selected_gyro = study.measurements(selected, 1, 0, True, True)
    selected_sensor = study.elevated_specific_force(selected, 1, 0, 0.6)
    interval = study.analysis_slice(100)
    trace_channels = {
        "ay_roll_axis_truth_mps2": decimate(
            selected["ay_truth"], base_rate, 100, True)[interval],
        "ay_roll_axis_reference_mps2": decimate(
            selected_ay, base_rate, 100, True)[interval],
        "ay_specific_force_h060_mps2": decimate(
            selected_sensor, base_rate, 100, True)[interval],
        "roll_rate_truth_radps": decimate(
            selected["roll_rate_truth"], base_rate, 100, True)[interval],
        "roll_rate_gyro_radps": decimate(
            selected_gyro, base_rate, 100, True)[interval],
    }
    trace_time = (float(config["analysis_start_s"])
                  + np.arange(len(trace_channels["ay_roll_axis_truth_mps2"]))
                  / 100.0)
    trace_rows = [{
        "time_s": trace_time[index],
        **{name: values[index] for name, values in trace_channels.items()},
    } for index in range(len(trace_time))]
    write_csv(DATA / "selected_trace_seed001_100Hz.csv", trace_rows)

    # Figures.
    rates = np.asarray(config["sampling_rates_hz"], dtype=float)
    rate_median, rate_p90 = [], []
    for rate in rates:
        group = [row for row in rate_rows
                 if int(row["sample_rate_hz"]) == int(rate)
                 and row["preprocessing"] == "antialiased"]
        values = np.asarray([[row["abs_I_error"], row["abs_C_error"],
                              row["abs_K_error"]] for row in group])
        rate_median.append(np.median(values, axis=0))
        rate_p90.append(np.percentile(values, 90.0, axis=0))
    rate_median = np.asarray(rate_median)
    rate_p90 = np.asarray(rate_p90)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for index, label, color in ((0, r"$I_{xx}$", "#1f77b4"),
                                (1, r"$C_\phi$", "#d62728"),
                                (2, r"$K_{\rm eff}$", "#2ca02c")):
        ax.semilogx(rates, rate_median[:, index], marker="o", color=color,
                    label=f"{label} median")
        ax.semilogx(rates, rate_p90[:, index], linestyle="--", marker=".",
                    color=color, alpha=0.75, label=f"{label} P90")
    ax.axvline(2.0, color="black", linestyle=":", linewidth=1.0)
    ax.text(2.05, ax.get_ylim()[1] * 0.86, "7 retained lines",
            fontsize=8, rotation=90, va="top")
    ax.set_xlabel("sampling rate (Hz)")
    ax.set_ylabel("absolute parameter error (%)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "sampling_rate_errors.pdf")
    fig.savefig(FIGURES / "sampling_rate_errors.png", dpi=220)
    plt.close(fig)

    heights = np.asarray(config["sensor_heights_m"], dtype=float)
    physical, apparent = [], []
    for height in heights:
        group = [row for row in placement_rows
                 if math.isclose(float(row["sensor_height_m"]), height)]
        physical.append(np.median([row["I_xx_est_kgm2"] for row in group]))
        apparent.append(p0.inertia - p0.mass * study.hr * height)
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.plot(heights, [p0.inertia] * len(heights), "k--", label="physical inertia")
    ax.plot(heights, apparent, "o-", label="analytical apparent inertia")
    ax.plot(heights, physical, "s", label="median fitted inertia")
    ax.set_xlabel("sensor height above roll axis (m)")
    ax.set_ylabel(r"inertia ($\mathrm{kg\,m^2}$)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "sensor_height_bias.pdf")
    fig.savefig(FIGURES / "sensor_height_bias.png", dpi=220)
    plt.close(fig)

    def median_errors(group: list[dict]) -> np.ndarray:
        return np.median(np.asarray([
            [row["abs_I_error"], row["abs_C_error"], row["abs_K_error"]]
            for row in group]), axis=0)

    excitation_values = np.vstack((
        median_errors(list(baseline_by_seed.values())),
        median_errors([row for row in excitation_rows
                       if row.get("excitation") == "low"])))
    road_values = np.vstack((
        median_errors(list(baseline_by_seed.values())),
        median_errors([row for row in road_rows
                       if row["road_condition"] == "dynamic_road_unmeasured"]),
        median_errors([row for row in road_rows
                       if row["road_condition"] == "dynamic_road_exact_static_proxy"]),
        median_errors([row for row in road_rows
                       if row["road_condition"] ==
                       "dynamic_road_noisy_delayed_static_proxy"])))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7))
    colors = ("#1f77b4", "#d62728", "#2ca02c")
    labels = (r"$I_{xx}$", r"$C_\phi$", r"$K_{\rm eff}$")
    width = 0.22
    x = np.arange(2)
    for index in range(3):
        axes[0].bar(x + (index - 1) * width, excitation_values[:, index],
                    width, color=colors[index], label=labels[index])
    axes[0].set_xticks(x, ("broadband", "low-band"))
    axes[0].set_ylabel("median absolute error (%)")
    axes[0].set_title("Matched excitation")
    axes[0].grid(True, axis="y", alpha=0.3)
    x = np.arange(4)
    for index in range(3):
        axes[1].bar(x + (index - 1) * width, road_values[:, index],
                    width, color=colors[index], label=labels[index])
    axes[1].set_xticks(
        x, ("zero road", "unmeasured", "exact proxy", "noisy/delayed"),
        rotation=15)
    axes[1].set_title("Dynamic road-moment mismatch")
    axes[1].grid(True, axis="y", alpha=0.3)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3,
               fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(FIGURES / "controlled_ablations.pdf")
    fig.savefig(FIGURES / "controlled_ablations.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    comparisons = ("P75-P0", "P150-P0")
    x = np.arange(len(comparisons))
    width = 0.24
    for offset, pairing, label, color in (
            (-width, "independent_acquisition", "independent simulated acquisition", "#2ca02c"),
            (0.0, "independent_white", "independent white noise", "#1f77b4"),
            (width, "common_acquisition", "exact common acquisition", "#ff7f0e")):
        medians = []
        p90s = []
        for comparison in comparisons:
            row = next(item for item in difference_summary
                       if item["pairing"] == pairing
                       and item["comparison"] == comparison)
            medians.append(row["median_abs_delta_I_error_pct"])
            p90s.append(row["p90_abs_delta_I_error_pct"])
        ax.bar(x + offset, medians, width, color=color, label=label)
        ax.scatter(x + offset, p90s, marker="_", s=180, color="black", zorder=3)
    ax.set_xticks(x, comparisons)
    ax.set_ylabel("inertia-change error (%)")
    ax.set_title("Payload acquisition pairing (bars: median; marks: P90)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "payload_noise_pairing.pdf")
    fig.savefig(FIGURES / "payload_noise_pairing.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for excitation, color in (("broad", "#1f77b4"), ("low", "#d62728")):
        group = [row for row in profile_rows if row["excitation"] == excitation]
        curves = []
        for seed in seeds:
            seed_rows = sorted(
                [row for row in group if int(row["seed"]) == seed],
                key=lambda item: item["I_xx_fixed_kgm2"])
            curves.append(np.interp(
                base_inertia_grid,
                [row["I_xx_fixed_kgm2"] for row in seed_rows],
                [row["relative_profile_objective"] for row in seed_rows]))
        curves = np.asarray(curves)
        median_curve = np.median(curves, axis=0)
        q10_curve = np.percentile(curves, 10.0, axis=0)
        q90_curve = np.percentile(curves, 90.0, axis=0)
        ax.semilogy(
            base_inertia_grid, median_curve,
            label=("broadband median" if excitation == "broad"
                   else "low-band median"),
            color=color)
        ax.fill_between(
            base_inertia_grid, q10_curve, q90_curve,
            color=color, alpha=0.18)
    ax.axvline(p0.inertia, color="black", linestyle="--", label="injected inertia")
    ax.set_xlabel(r"fixed inertia ($\mathrm{kg\,m^2}$)")
    ax.set_ylabel("profile objective / minimum")
    ax.set_title("Bounded profiled inertia objective across 30 seeds")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "inertia_profiles.pdf")
    fig.savefig(FIGURES / "inertia_profiles.png", dpi=220)
    plt.close(fig)

    metadata = {
        "study_version": config["study_version"],
        "physical_experiment": False,
        "generator_included": True,
        "estimator": "profiled_frequency_domain_errors_in_variables_with_controls",
        "distinct_full_parameter_fits": len(all_rows),
        "profile_reoptimizations": len(profile_rows),
        "seeds": len(list(seeds)),
        "payload_pairings": [
            "exact common acquisition", "independent white noise",
            "independent simulated acquisition phases/RMS/calibration/noise"],
        "payload_geometry": config["payload_geometry"],
        "profile_grid_points_per_seed_condition": profile_points,
        "road_proxy_sensitivity": {
            "noise_fraction_of_rms": config["road_proxy_noise_fraction_of_rms"],
            "delay_s": config["road_proxy_delay_s"],
        },
        "objective_normalization": (
            "one_over_Nf_complex_sum; equivalent to twice the mean squared "
            "stacked real-imaginary residual"),
        "selected_trace_preprocessing": (
            "same explicit antialias FIR and analysis interval as the 100 Hz "
            "rate-study condition"),
        "antialias_filter": "explicit 20*d+1 tap Kaiser(beta=5) FIR",
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (RESULTS / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    print(
        f"completed v5 study: {len(all_rows)} distinct full-parameter fits, "
        f"{len(profile_rows)} profile reoptimizations, {len(list(seeds))} seeds")


if __name__ == "__main__":
    run()