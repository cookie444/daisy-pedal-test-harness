"""Matplotlib figures for the HTML report (Agg backend, embedded as base64)."""

from __future__ import annotations

import base64
from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = [
    "figure_to_png",
    "plot_delay",
    "plot_decay",
    "plot_spectrum",
    "plot_response",
    "plot_stutter",
    "plot_waveform",
    "plot_latency",
]

COLOUR = {
    "input": "#7f8c9b",
    "output": "#2f6f9f",
    "marker": "#d1495b",
    "expected": "#2a9d8f",
    "grid": "#dcdcdc",
}


def figure_to_png(fig: "matplotlib.figure.Figure", dpi: int = 90) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _style(ax: "matplotlib.axes.Axes", title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, color=COLOUR["grid"], linewidth=0.6)
    ax.tick_params(labelsize=7)


def plot_delay(
    capture: np.ndarray,
    sr: int,
    details: dict,
    *,
    expected_ms: float,
    title: str = "Delay capture",
    max_ms: float | None = None,
) -> str:
    t = np.arange(len(capture)) / sr * 1000.0
    times = details.get("times_ms", np.array([]))
    limit = max_ms or (max(times[-1] * 1.15, expected_ms * 4) if times.size else 1000.0)
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    ax.plot(t, capture, color=COLOUR["output"], linewidth=0.6, label="pedal out")
    if details.get("dry_time_ms") is not None:
        ax.axvline(details["dry_time_ms"], color=COLOUR["marker"], linestyle=":", linewidth=1.0, label="dry")
    for i, (time_ms, amp) in enumerate(zip(times, details.get("amps", []))):
        ax.plot(time_ms, amp, "v", color=COLOUR["marker"], markersize=4)
        if i < 12:
            ax.annotate(f"{i + 1}", (time_ms, amp), fontsize=6, ha="center", va="bottom")
    origin = details.get("dry_time_ms") or 0.0
    for k in range(1, 5):
        ax.axvline(origin + k * expected_ms, color=COLOUR["expected"], alpha=0.35, linewidth=0.8)
    ax.set_xlim(0, limit)
    _style(ax, title, "time (ms)", "amplitude")
    ax.legend(fontsize=7, loc="upper right")
    return figure_to_png(fig)


def plot_decay(
    details: dict,
    *,
    expected_db: float,
    tol_db: float,
    slope_db: float,
    title: str = "Repeat decay",
) -> str:
    amps = np.asarray(details.get("amps", []), dtype=float)
    fig, ax = plt.subplots(figsize=(5.4, 2.4))
    if amps.size:
        db = 20 * np.log10(np.maximum(amps, 1e-12))
        db_rel = db - db[0]
        index = np.arange(len(db_rel))
        ax.plot(index, db_rel, "o-", color=COLOUR["output"], markersize=3.5, label="measured repeats")
        fit = np.polyfit(index, db_rel, 1)
        ax.plot(
            index,
            np.polyval(fit, index),
            "--",
            color=COLOUR["marker"],
            linewidth=1.0,
            label=f"fit: {slope_db:.2f} dB/repeat",
        )
        ax.axhline(-60, color="black", linewidth=0.8, alpha=0.5)
        ax.annotate("-60 dB", (index[-1] * 0.02, -58), fontsize=6)
    ax.axhspan(expected_db - tol_db, expected_db + tol_db, color=COLOUR["expected"], alpha=0.12, label="spec window")
    ax.axhline(expected_db, color=COLOUR["expected"], linewidth=0.9)
    _style(ax, title, "repeat #", "level vs first repeat (dB)")
    ax.legend(fontsize=6, loc="upper right")
    return figure_to_png(fig)


def plot_spectrum(details: dict, *, f0: float, title: str = "Distortion spectrum") -> str:
    freqs = details.get("freqs")
    mag_db = details.get("mag_db")
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    if freqs is not None and mag_db is not None:
        peak = float(np.max(mag_db))
        ax.plot(freqs, mag_db - peak, color=COLOUR["output"], linewidth=0.7)
        ax.set_xlim(0, min(max(8 * f0, 1000), freqs[-1]))
        ax.set_ylim(-120, 5)
        for k, level in enumerate(details.get("harmonics_db", []), start=2):
            if np.isfinite(level) and level > -110:
                ax.plot(k * f0, level, "v", color=COLOUR["marker"], markersize=4)
                ax.annotate(f"H{k}", (k * f0, level), fontsize=6, ha="center", va="bottom")
    _style(ax, title, "frequency (Hz)", "level (dB re fundamental)")
    return figure_to_png(fig)


def plot_response(
    details: dict,
    *,
    ripple_limit_db: float,
    band: tuple[float, float] = (100.0, 10000.0),
    title: str = "Frequency response",
) -> str:
    grid = details.get("grid_hz")
    resp = details.get("resp_db")
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    if grid is not None and resp is not None:
        ax.semilogx(grid, resp, color=COLOUR["output"], linewidth=1.1, label="measured")
        mid = details.get("gain_1khz_db", 0.0)
        ax.axhspan(mid - ripple_limit_db / 2, mid + ripple_limit_db / 2, color=COLOUR["expected"], alpha=0.12, label=f"+/-{ripple_limit_db / 2:g} dB mask")
        ax.axvspan(band[0], band[1], color="black", alpha=0.05)
        for corner, label in ((details.get("corner_low_hz"), "low"), (details.get("corner_high_hz"), "high")):
            if corner and np.isfinite(corner):
                ax.axvline(corner, color=COLOUR["marker"], linestyle="--", linewidth=0.9)
                ax.annotate(f"{label} -3 dB\n{corner:.0f} Hz", (corner, mid - 1), fontsize=6)
        ax.set_xlim(grid[0], grid[-1])
        pad = max(ripple_limit_db, 6.0)
        ax.set_ylim(mid - 3 * pad, mid + pad)
    _style(ax, title, "frequency (Hz)", "gain (dB)")
    ax.legend(fontsize=6, loc="lower left")
    return figure_to_png(fig)


def plot_stutter(capture: np.ndarray, sr: int, details: dict, *, title: str = "Stutter capture") -> str:
    t = np.arange(len(capture)) / sr * 1000.0
    trig = details.get("trigger_s", 0.0) * 1000.0
    slice_ms = details.get("slice_ms", 0.0)
    gate_ms = details.get("gate_ms", 0.0)
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    ax.plot(t, capture, color=COLOUR["output"], linewidth=0.5, label="pedal out")
    ax.axvline(trig, color=COLOUR["marker"], linewidth=1.0, label="trigger")
    ax.axvspan(trig, trig + gate_ms, color=COLOUR["marker"], alpha=0.08, label=f"gate {gate_ms:.1f} ms")
    for k in range(1, 8):
        edge = trig + k * slice_ms
        if edge > trig + gate_ms + slice_ms:
            break
        ax.axvline(edge, color=COLOUR["expected"], linewidth=0.7, alpha=0.7)
    ax.set_xlim(max(0.0, trig - 150), trig + gate_ms + 250)
    _style(ax, title, "time (ms)", "amplitude")
    ax.legend(fontsize=6, loc="upper right")
    return figure_to_png(fig)


def plot_waveform(capture: np.ndarray, sr: int, *, title: str, ylabel: str = "amplitude") -> str:
    t = np.arange(len(capture)) / sr * 1000.0
    fig, ax = plt.subplots(figsize=(6.0, 2.0))
    ax.plot(t, capture, color=COLOUR["output"], linewidth=0.5)
    _style(ax, title, "time (ms)", ylabel)
    return figure_to_png(fig)


def plot_latency(
    bypass: np.ndarray, active: np.ndarray, sr: int, *, title: str = "Latency: bypass vs active"
) -> str:
    span = int(0.02 * sr)
    b = bypass[:span] / (np.max(np.abs(bypass)) or 1.0)
    a = active[:span] / (np.max(np.abs(active)) or 1.0)
    t = np.arange(span) / sr * 1000.0
    fig, ax = plt.subplots(figsize=(5.6, 2.2))
    ax.plot(t, b, color=COLOUR["input"], linewidth=1.0, label="bypassed")
    ax.plot(t, a, color=COLOUR["output"], linewidth=1.0, label="through pedal")
    _style(ax, title, "time (ms)", "normalised amplitude")
    ax.legend(fontsize=6, loc="upper right")
    return figure_to_png(fig)
