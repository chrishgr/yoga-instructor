"""Audio feedback — maps a scalar deviation to a continuous tone.

Runs on a background thread so it never blocks the main frame loop.
Lower deviation -> higher, cleaner tone. Higher deviation -> lower, less
stable tone. The mapping is deliberately simple; it can be replaced with
something richer (e.g. multi-voice chords) later.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np


class AudioFeedback:
    def __init__(
        self,
        base_frequency: float = 220.0,
        max_frequency: float = 880.0,
        max_deviation_deg: float = 45.0,
        sample_rate: int = 44100,
        enabled: bool = True,
    ) -> None:
        self.base_frequency = base_frequency
        self.max_frequency = max_frequency
        self.max_deviation_deg = max_deviation_deg
        self.sample_rate = sample_rate
        self.enabled = enabled

        self._current_deviation = float("nan")
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._pygame_sound = None

    # -- Public API --------------------------------------------------

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update(self, deviation: float) -> None:
        with self._lock:
            self._current_deviation = deviation

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        try:
            import pygame

            pygame.mixer.quit()
        except Exception:
            pass

    # -- Background loop ---------------------------------------------

    def _run(self) -> None:
        try:
            import pygame
        except ImportError:
            print("[audio] pygame not installed — audio feedback disabled")
            return

        pygame.mixer.pre_init(frequency=self.sample_rate, size=-16, channels=1)
        pygame.mixer.init()

        chunk_duration = 0.1  # seconds per regenerated chunk
        phase = 0.0

        while not self._stop_event.is_set():
            with self._lock:
                deviation = self._current_deviation

            freq = self._deviation_to_frequency(deviation)
            samples, phase = self._generate_tone(freq, chunk_duration, phase)
            sound = pygame.sndarray.make_sound(samples)
            sound.play()
            time.sleep(chunk_duration)

    def _deviation_to_frequency(self, deviation: float) -> float:
        if math.isnan(deviation):
            return self.base_frequency
        # Clamp and invert: 0 deg -> max freq, max_deviation -> base freq.
        clamped = max(0.0, min(deviation, self.max_deviation_deg))
        ratio = 1.0 - (clamped / self.max_deviation_deg)
        return self.base_frequency + ratio * (self.max_frequency - self.base_frequency)

    def _generate_tone(
        self, frequency: float, duration: float, phase: float
    ) -> tuple[np.ndarray, float]:
        n_samples = int(self.sample_rate * duration)
        t = np.arange(n_samples) / self.sample_rate
        wave = np.sin(2 * np.pi * frequency * t + phase)
        # Update phase so successive chunks are continuous.
        phase = (phase + 2 * np.pi * frequency * duration) % (2 * np.pi)
        samples = (wave * 0.2 * 32767).astype(np.int16)
        return samples, phase
