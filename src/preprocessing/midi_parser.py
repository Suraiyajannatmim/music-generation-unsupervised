"""MIDI discovery and genre helpers."""
from pathlib import Path

MIDI_SUFFIXES = {".mid", ".midi"}


def find_midi_files(root):
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in MIDI_SUFFIXES)


def infer_genre_from_path(midi_path, raw_root):
    midi_path = Path(midi_path)
    raw_root = Path(raw_root)
    try:
        rel = midi_path.relative_to(raw_root)
        return rel.parts[0].lower() if len(rel.parts) > 1 else "unknown"
    except ValueError:
        return "unknown"
