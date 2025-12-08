"""Audio utility helpers for encoding/decoding."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def load_wav(path: Path) -> Tuple[np.ndarray, int]:
    """Load a WAV file into a numpy array.

    TODO: Use soundfile or librosa to implement loading while keeping the
    dependency footprint manageable.
    """

    raise NotImplementedError("Audio loading not yet implemented")


def save_wav(samples: np.ndarray, sample_rate: int, path: Path) -> Path:
    """Save audio samples to a WAV file.

    TODO: Implement writing via soundfile or wave module.
    """

    raise NotImplementedError("Audio saving not yet implemented")
