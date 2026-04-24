from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_MIDI_DIR = DATA_DIR / "raw_midi"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
GENERATED_MIDI_DIR = OUTPUT_DIR / "generated_midis"
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"
RLHF_DIR = OUTPUT_DIR / "rlhf"

GENRES = ["classical", "jazz", "pop", "rock", "electronic"]
SEED = 42
SEQ_LEN_AE = 128
SEQ_LEN_TRANSFORMER = 256
BATCH_SIZE = 32
