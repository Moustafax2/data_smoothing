"""Pluggable smoothing filters."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.signal import savgol_filter

Smoother = Callable[[np.ndarray], np.ndarray]


def ema_smoother(alpha: float = 0.2) -> Smoother:
    """Build an EMA smoother with a fixed alpha."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("EMA alpha must be in the range (0, 1].")

    def smooth(values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=float)
        if data.size == 0:
            return data.copy()

        smoothed = np.empty_like(data, dtype=float)
        smoothed[0] = data[0]
        for idx in range(1, len(data)):
            smoothed[idx] = alpha * data[idx] + (1.0 - alpha) * smoothed[idx - 1]
        return smoothed

    return smooth


def savgol_smoother(window_length: int = 50, polyorder: int = 2) -> Smoother:
    """Build a Savitzky-Golay smoother for offline benchmarking."""
    if window_length < 3 or window_length % 2 == 0:
        raise ValueError("Savitzky-Golay window_length must be an odd integer >= 3.")
    if polyorder < 1 or polyorder >= window_length:
        raise ValueError("Savitzky-Golay polyorder must be >= 1 and < window_length.")

    def smooth(values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=float)
        if data.size == 0:
            return data.copy()

        adjusted_window = min(window_length, data.size if data.size % 2 == 1 else data.size - 1)
        if adjusted_window < 3:
            return data.copy()
        adjusted_polyorder = min(polyorder, adjusted_window - 1)
        return savgol_filter(data, window_length=adjusted_window, polyorder=adjusted_polyorder, mode="interp")

    return smooth


def one_euro_smoother(
    min_cutoff: float = 0.35,
    beta: float = 0.008,
    derivative_cutoff: float = 0.6,
    frame_rate: float = 30.0,
) -> Smoother:
    """Build a One Euro filter for real-time smoothing with adaptive lag."""
    if min_cutoff <= 0.0:
        raise ValueError("One Euro min_cutoff must be > 0.")
    if beta < 0.0:
        raise ValueError("One Euro beta must be >= 0.")
    if derivative_cutoff <= 0.0:
        raise ValueError("One Euro derivative_cutoff must be > 0.")
    if frame_rate <= 0.0:
        raise ValueError("One Euro frame_rate must be > 0.")

    def alpha_from_cutoff(cutoff: float) -> float:
        dt = 1.0 / frame_rate
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def smooth(values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=float)
        if data.size == 0:
            return data.copy()

        smoothed = np.empty_like(data, dtype=float)
        smoothed[0] = data[0]
        prev_x = data[0]
        prev_dx = 0.0

        derivative_alpha = alpha_from_cutoff(derivative_cutoff)

        for idx in range(1, len(data)):
            dx = (data[idx] - prev_x) * frame_rate
            dx_hat = derivative_alpha * dx + (1.0 - derivative_alpha) * prev_dx
            cutoff = min_cutoff + beta * abs(dx_hat)
            signal_alpha = alpha_from_cutoff(cutoff)
            smoothed[idx] = signal_alpha * data[idx] + (1.0 - signal_alpha) * smoothed[idx - 1]
            prev_x = data[idx]
            prev_dx = dx_hat

        return smoothed

    return smooth


def alpha_beta_smoother(
    alpha: float = 0.18,
    beta: float = 0.004,
    dt: float = 1.0,
) -> Smoother:
    """Build a 1D alpha-beta tracker with position and velocity state."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("Alpha-beta alpha must be in the range (0, 1].")
    if beta < 0.0:
        raise ValueError("Alpha-beta beta must be >= 0.")
    if dt <= 0.0:
        raise ValueError("Alpha-beta dt must be > 0.")

    def smooth(values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=float)
        if data.size == 0:
            return data.copy()

        smoothed = np.empty_like(data, dtype=float)
        position = data[0]
        velocity = 0.0
        smoothed[0] = position

        for idx in range(1, len(data)):
            predicted_position = position + velocity * dt
            residual = data[idx] - predicted_position
            position = predicted_position + alpha * residual
            velocity = velocity + (beta / dt) * residual
            smoothed[idx] = position

        return smoothed

    return smooth


def pid_smoother(
    kp: float = 0.18,
    ki: float = 0.002,
    kd: float = 0.08,
    integral_limit: float = 25.0,
    dt: float = 1.0,
) -> Smoother:
    """Build a simple PID-style tracker that chases the measured point."""
    if kp < 0.0 or ki < 0.0 or kd < 0.0:
        raise ValueError("PID gains must be >= 0.")
    if integral_limit <= 0.0:
        raise ValueError("PID integral_limit must be > 0.")
    if dt <= 0.0:
        raise ValueError("PID dt must be > 0.")

    def smooth(values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=float)
        if data.size == 0:
            return data.copy()

        smoothed = np.empty_like(data, dtype=float)
        output = data[0]
        integral = 0.0
        previous_error = 0.0
        smoothed[0] = output

        for idx in range(1, len(data)):
            error = data[idx] - output
            integral = np.clip(integral + error * dt, -integral_limit, integral_limit)
            derivative = (error - previous_error) / dt
            control = kp * error + ki * integral + kd * derivative
            output = output + control
            smoothed[idx] = output
            previous_error = error

        return smoothed

    return smooth


def hybrid_realtime_smoother(
    outlier_window: int = 7,
    outlier_gate: float = 2.2,
    cv_process_variance: float = 7e-2,
    cv_measurement_variance: float = 18.0,
    base_smoothing: float = 0.68,
) -> Smoother:
    """Robust adaptive real-time smoother tuned for low jitter and moderate lag."""
    if outlier_window < 3:
        raise ValueError("Hybrid outlier_window must be >= 3.")
    if outlier_gate <= 0.0:
        raise ValueError("Hybrid outlier_gate must be > 0.")
    if cv_process_variance <= 0.0:
        raise ValueError("Hybrid cv_process_variance must be > 0.")
    if cv_measurement_variance <= 0.0:
        raise ValueError("Hybrid cv_measurement_variance must be > 0.")
    if not 0.0 < base_smoothing < 1.0:
        raise ValueError("Hybrid base_smoothing must be in the range (0, 1).")

    def smooth(values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=float)
        if data.size == 0:
            return data.copy()

        dt = 1.0
        transition = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
        measurement_matrix = np.array([[1.0, 0.0]], dtype=float)
        process_noise = cv_process_variance * np.array(
            [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]],
            dtype=float,
        )
        base_measurement_noise = np.array([[cv_measurement_variance]], dtype=float)
        identity = np.eye(2, dtype=float)

        state = np.array([[data[0]], [0.0]], dtype=float)
        covariance = np.eye(2, dtype=float)
        smoothed = np.empty_like(data, dtype=float)
        smoothed[0] = data[0]

        for idx in range(1, len(data)):
            predicted_state = transition @ state
            predicted_covariance = transition @ covariance @ transition.T + process_noise

            recent = data[max(0, idx - outlier_window) : idx]
            if recent.size == 0:
                median = data[idx - 1]
                sigma = cv_measurement_variance**0.5
            else:
                median = float(np.median(recent))
                mad = float(np.median(np.abs(recent - median)))
                sigma = 1.4826 * mad + 1e-6

            clipped_measurement = median + np.clip(
                data[idx] - median,
                -outlier_gate * sigma,
                outlier_gate * sigma,
            )
            measurement = np.array([[clipped_measurement]], dtype=float)
            innovation = measurement - measurement_matrix @ predicted_state

            innovation_scale = abs(float(innovation[0, 0])) / (2.5 * sigma + 1e-6)
            adaptive_measurement_noise = base_measurement_noise * (1.0 + min(innovation_scale, 6.0))

            if abs(float(innovation[0, 0])) > 6.5 * sigma and idx > outlier_window:
                state = predicted_state
                covariance = predicted_covariance
                estimate = float(state[0, 0])
            else:
                innovation_covariance = (
                    measurement_matrix @ predicted_covariance @ measurement_matrix.T
                    + adaptive_measurement_noise
                )
                kalman_gain = (
                    predicted_covariance
                    @ measurement_matrix.T
                    @ np.linalg.inv(innovation_covariance)
                )
                state = predicted_state + kalman_gain @ innovation
                covariance = (identity - kalman_gain @ measurement_matrix) @ predicted_covariance
                estimate = float(state[0, 0])

            motion_ratio = min(abs(float(state[1, 0])) / 4.0, 1.0)
            smoothing_alpha = min(base_smoothing + 0.12 * motion_ratio, 0.9)
            smoothed[idx] = smoothing_alpha * estimate + (1.0 - smoothing_alpha) * smoothed[idx - 1]

        return smoothed

    return smooth


def fixed_lag_adaptive_smoother(
    lag_frames: int = 4,
    polyorder: int = 2,
) -> Smoother:
    """Refine the adaptive real-time filter with a tiny fixed delay window."""
    if lag_frames < 1:
        raise ValueError("Fixed-lag lag_frames must be >= 1.")
    if polyorder < 1:
        raise ValueError("Fixed-lag polyorder must be >= 1.")

    base_smoother = hybrid_realtime_smoother()

    def smooth(values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=float)
        if data.size == 0:
            return data.copy()

        base = base_smoother(data)
        window_length = min(2 * lag_frames + 1, len(base) if len(base) % 2 == 1 else len(base) - 1)
        if window_length < 3:
            return base

        adjusted_polyorder = min(polyorder, window_length - 1)
        refined = savgol_filter(
            base,
            window_length=window_length,
            polyorder=adjusted_polyorder,
            mode="interp",
        )

        # Preserve the tail where the fixed-lag system would not yet have enough future frames.
        output = refined.copy()
        output[-lag_frames:] = base[-lag_frames:]
        return output

    return smooth


def kalman_smoother(
    process_variance: float = 1e-2,
    measurement_variance: float = 20.0,
) -> Smoother:
    """Build a simple 1D constant-position Kalman filter."""
    if process_variance <= 0.0:
        raise ValueError("Kalman process_variance must be > 0.")
    if measurement_variance <= 0.0:
        raise ValueError("Kalman measurement_variance must be > 0.")

    def smooth(values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=float)
        if data.size == 0:
            return data.copy()

        estimate = data[0]
        error_covariance = 1.0
        smoothed = np.empty_like(data, dtype=float)
        smoothed[0] = estimate

        for idx in range(1, len(data)):
            error_covariance += process_variance
            kalman_gain = error_covariance / (error_covariance + measurement_variance)
            estimate = estimate + kalman_gain * (data[idx] - estimate)
            error_covariance = (1.0 - kalman_gain) * error_covariance
            smoothed[idx] = estimate

        return smoothed

    return smooth


def constant_velocity_kalman_smoother(
    process_variance: float = 1e-2,
    measurement_variance: float = 20.0,
    dt: float = 1.0,
) -> Smoother:
    """Build a 1D constant-velocity Kalman filter."""
    if process_variance <= 0.0:
        raise ValueError("Constant-velocity Kalman process_variance must be > 0.")
    if measurement_variance <= 0.0:
        raise ValueError("Constant-velocity Kalman measurement_variance must be > 0.")
    if dt <= 0.0:
        raise ValueError("Constant-velocity Kalman dt must be > 0.")

    def smooth(values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=float)
        if data.size == 0:
            return data.copy()

        state = np.array([[data[0]], [0.0]], dtype=float)
        covariance = np.eye(2, dtype=float)
        transition = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
        measurement_matrix = np.array([[1.0, 0.0]], dtype=float)
        process_noise = process_variance * np.array(
            [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]],
            dtype=float,
        )
        measurement_noise = np.array([[measurement_variance]], dtype=float)
        identity = np.eye(2, dtype=float)

        smoothed = np.empty_like(data, dtype=float)
        smoothed[0] = state[0, 0]

        for idx in range(1, len(data)):
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process_noise

            measurement = np.array([[data[idx]]], dtype=float)
            innovation = measurement - measurement_matrix @ state
            innovation_covariance = measurement_matrix @ covariance @ measurement_matrix.T + measurement_noise
            kalman_gain = covariance @ measurement_matrix.T @ np.linalg.inv(innovation_covariance)

            state = state + kalman_gain @ innovation
            covariance = (identity - kalman_gain @ measurement_matrix) @ covariance
            smoothed[idx] = state[0, 0]

        return smoothed

    return smooth


def available_smoothers(
    alpha: float = 0.2,
    savgol_window_length: int = 50,
    savgol_polyorder: int = 2,
    one_euro_min_cutoff: float = 0.35,
    one_euro_beta: float = 0.008,
    one_euro_derivative_cutoff: float = 0.6,
    one_euro_frame_rate: float = 30.0,
    alpha_beta_alpha: float = 0.18,
    alpha_beta_beta: float = 0.004,
    pid_kp: float = 0.18,
    pid_ki: float = 0.002,
    pid_kd: float = 0.08,
    pid_integral_limit: float = 25.0,
    constant_velocity_process_variance: float = 1e-2,
    constant_velocity_measurement_variance: float = 20.0,
    fixed_lag_frames: int = 4,
) -> dict[str, Smoother]:
    """Registry for current and future smoothing methods."""
    return {
        "ema": ema_smoother(alpha=alpha),
        "savgol": savgol_smoother(
            window_length=savgol_window_length,
            polyorder=savgol_polyorder,
        ),
        "one_euro": one_euro_smoother(
            min_cutoff=one_euro_min_cutoff,
            beta=one_euro_beta,
            derivative_cutoff=one_euro_derivative_cutoff,
            frame_rate=one_euro_frame_rate,
        ),
        "alpha_beta": alpha_beta_smoother(
            alpha=alpha_beta_alpha,
            beta=alpha_beta_beta,
        ),
        "pid": pid_smoother(
            kp=pid_kp,
            ki=pid_ki,
            kd=pid_kd,
            integral_limit=pid_integral_limit,
        ),
        "cv_kalman": constant_velocity_kalman_smoother(
            process_variance=constant_velocity_process_variance,
            measurement_variance=constant_velocity_measurement_variance,
        ),
        "hybrid_realtime": hybrid_realtime_smoother(
            outlier_window=7,
            outlier_gate=2.2,
            cv_process_variance=max(7e-2, constant_velocity_process_variance),
            cv_measurement_variance=min(18.0, constant_velocity_measurement_variance),
            base_smoothing=0.68,
        ),
        "fixed_lag_adaptive": fixed_lag_adaptive_smoother(
            lag_frames=fixed_lag_frames,
            polyorder=2,
        ),
    }


def smooth_series(values: np.ndarray, method: str = "ema", **kwargs: float) -> np.ndarray:
    """Dispatch to a named smoother."""
    registry = available_smoothers(
        alpha=kwargs.get("alpha", 0.2),
        savgol_window_length=int(kwargs.get("savgol_window_length", 51)),
        savgol_polyorder=int(kwargs.get("savgol_polyorder", 2)),
        one_euro_min_cutoff=kwargs.get("one_euro_min_cutoff", 1.0),
        one_euro_beta=kwargs.get("one_euro_beta", 0.02),
        one_euro_derivative_cutoff=kwargs.get("one_euro_derivative_cutoff", 1.0),
        one_euro_frame_rate=kwargs.get("one_euro_frame_rate", 30.0),
        alpha_beta_alpha=kwargs.get("alpha_beta_alpha", 0.18),
        alpha_beta_beta=kwargs.get("alpha_beta_beta", 0.004),
        pid_kp=kwargs.get("pid_kp", 0.18),
        pid_ki=kwargs.get("pid_ki", 0.002),
        pid_kd=kwargs.get("pid_kd", 0.08),
        pid_integral_limit=kwargs.get("pid_integral_limit", 25.0),
        constant_velocity_process_variance=kwargs.get("constant_velocity_process_variance", 1e-2),
        constant_velocity_measurement_variance=kwargs.get("constant_velocity_measurement_variance", 20.0),
        fixed_lag_frames=int(kwargs.get("fixed_lag_frames", 4)),
    )
    try:
        smoother = registry[method]
    except KeyError as exc:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unknown smoothing method '{method}'. Available: {available}") from exc
    return smoother(values)


def smooth_all_series(values: np.ndarray, **kwargs: float) -> dict[str, np.ndarray]:
    """Run every registered smoother on the same input series."""
    registry = available_smoothers(
        alpha=kwargs.get("alpha", 0.2),
        savgol_window_length=int(kwargs.get("savgol_window_length", 51)),
        savgol_polyorder=int(kwargs.get("savgol_polyorder", 2)),
        one_euro_min_cutoff=kwargs.get("one_euro_min_cutoff", 1.0),
        one_euro_beta=kwargs.get("one_euro_beta", 0.02),
        one_euro_derivative_cutoff=kwargs.get("one_euro_derivative_cutoff", 1.0),
        one_euro_frame_rate=kwargs.get("one_euro_frame_rate", 30.0),
        alpha_beta_alpha=kwargs.get("alpha_beta_alpha", 0.18),
        alpha_beta_beta=kwargs.get("alpha_beta_beta", 0.004),
        pid_kp=kwargs.get("pid_kp", 0.18),
        pid_ki=kwargs.get("pid_ki", 0.002),
        pid_kd=kwargs.get("pid_kd", 0.08),
        pid_integral_limit=kwargs.get("pid_integral_limit", 25.0),
        constant_velocity_process_variance=kwargs.get("constant_velocity_process_variance", 1e-2),
        constant_velocity_measurement_variance=kwargs.get("constant_velocity_measurement_variance", 20.0),
        fixed_lag_frames=int(kwargs.get("fixed_lag_frames", 4)),
    )
    return {name: smoother(values) for name, smoother in registry.items()}
