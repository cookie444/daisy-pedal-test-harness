"""Out-of-band pedal control.

Real hardware needs more than audio: tap tempo, bypass and the stutter
footswitch are events, not signals. ``SerialTransport`` speaks a tiny text
protocol over USB CDC (``TAP``, ``BYPASS 1``, ``STUTTER``) which is what a
Daisy firmware debug build would expose; ``NullTransport`` is used by the
simulator and the file backend.
"""

from __future__ import annotations

import time
from typing import Any

from .base import NullTransport, Transport

__all__ = ["NullTransport", "SerialTransport", "ScheduledTransport", "make_transport"]


class SerialTransport(Transport):
    """Text-line control over a serial port (pyserial, imported lazily)."""

    name = "serial"

    def __init__(self, port: str, baudrate: int = 115200, timeout_s: float = 1.0) -> None:
        import serial  # type: ignore[import-not-found]

        self.port = port
        self._serial: Any = serial.Serial(port, baudrate=baudrate, timeout=timeout_s)
        time.sleep(0.2)  # CDC ACM enumeration / bootloader delay

    def _write(self, line: str) -> None:
        self._serial.write(f"{line}\n".encode("ascii"))
        self._serial.flush()

    def tap(self) -> None:
        self._write("TAP")

    def set_bypass(self, enabled: bool) -> None:
        self._write(f"BYPASS {1 if enabled else 0}")

    def trigger_stutter(self) -> None:
        self._write("STUTTER")

    def close(self) -> None:
        try:
            self._serial.close()
        except Exception:  # pragma: no cover - best effort
            pass


class ScheduledTransport(Transport):
    """Fires ``trigger_stutter`` a fixed time after arming.

    Used by the audio backend: playback and recording start together, so a
    stutter can be scheduled against wall clock time.
    """

    name = "scheduled"

    def __init__(self, inner: Transport | None = None) -> None:
        self._inner = inner or NullTransport()
        self._timer: Any = None

    def schedule(self, delay_s: float) -> None:
        from threading import Timer

        self._timer = Timer(delay_s, self.trigger_stutter)
        self._timer.daemon = True
        self._timer.start()

    def cancel(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def tap(self) -> None:
        self._inner.tap()

    def set_bypass(self, enabled: bool) -> None:
        self._inner.set_bypass(enabled)

    def trigger_stutter(self) -> None:
        self._inner.trigger_stutter()

    def close(self) -> None:
        self.cancel()
        self._inner.close()


def make_transport(name: str, **kwargs: Any) -> Transport:
    if name in ("null", "", None):
        return NullTransport()
    if name == "serial":
        port = kwargs.get("port")
        if not port:
            raise ValueError("serial transport requires a port")
        return SerialTransport(str(port), int(kwargs.get("baudrate", 115200)))
    raise ValueError(f"unknown transport '{name}'")
