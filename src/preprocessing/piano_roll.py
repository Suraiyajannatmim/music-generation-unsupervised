"""Optional piano-roll helpers for report extensions."""
import numpy as np


def empty_piano_roll(time_steps=128, pitches=128):
    return np.zeros((time_steps, pitches), dtype=np.float32)
