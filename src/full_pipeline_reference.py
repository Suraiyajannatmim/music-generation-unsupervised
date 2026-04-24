import os

import gc

import copy

import json

import math

import random

from pathlib import Path

from collections import Counter, defaultdict



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm



import pretty_midi



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader



device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

PROJECT_ROOT = Path("/content/drive/MyDrive/music_generation_project")



RAW_MIDI_DIR = PROJECT_ROOT / "data" / "raw_midi"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

GENERATED_DIR = OUTPUT_DIR / "generated_midis"

PLOT_DIR = OUTPUT_DIR / "plots"

TABLE_DIR = OUTPUT_DIR / "tables"

RLHF_DIR = OUTPUT_DIR / "rlhf"



for p in [RAW_MIDI_DIR, PROCESSED_DIR, OUTPUT_DIR, GENERATED_DIR, PLOT_DIR, TABLE_DIR, RLHF_DIR]:

    p.mkdir(parents=True, exist_ok=True)



print("Project root:", PROJECT_ROOT)

print("Raw MIDI directory:", RAW_MIDI_DIR)
# Reproducibility

SEED = 42

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)



# Preprocessing

MAX_FILES = None            # Set to a small integer like 200 for quick debugging

SKIP_DRUMS = True

USE_ONLY_PIANO = False      # Good for MAESTRO-style experiments

MIN_NOTES_PER_FILE = 10



# Tokenization

TIME_SHIFT_STEP = 0.02      # seconds

MAX_TIME_SHIFT = 1.00       # seconds

VELOCITY_STEP = 4



# Sequence lengths

SEQ_LEN_AE_VAE = 128

SEQ_LEN_TRANSFORMER = 256

WINDOW_STRIDE = 64



# Data split

TRAIN_RATIO = 0.80

VAL_RATIO = 0.10

TEST_RATIO = 0.10



# Dataloaders

BATCH_SIZE = 32

NUM_WORKERS = 2



# Task 1: LSTM Autoencoder

T1_EMB_DIM = 256

T1_HIDDEN_DIM = 512

T1_LATENT_DIM = 128

T1_NUM_LAYERS = 2

T1_LR = 1e-3

T1_EPOCHS = 15

T1_SINGLE_GENRE = None      # None => auto-pick first non-unknown genre or most common



# Task 2: VAE

T2_EMB_DIM = 256

T2_HIDDEN_DIM = 512

T2_LATENT_DIM = 128

T2_NUM_LAYERS = 2

T2_GENRE_EMB_DIM = 32

T2_LR = 1e-3

T2_EPOCHS = 20

T2_BETA = 0.1



# Task 3: Transformer

T3_EMB_DIM = 256

T3_NHEAD = 8

T3_NUM_LAYERS = 4

T3_FF_DIM = 1024

T3_DROPOUT = 0.1

T3_GENRE_EMB_DIM = 256

T3_LR = 1e-4

T3_EPOCHS = 15

T3_MAX_GEN_LEN = 512



# Task 4: RLHF

T4_CANDIDATES_PER_GENRE = 3

T4_REWARD_HIDDEN = 64

T4_REWARD_EPOCHS = 50

T4_REWARD_LR = 1e-3

T4_RL_STEPS = 50

T4_RL_BATCH_SIZE = 8

T4_RL_LR = 5e-5

T4_RL_MAX_LEN = 256
TIME_SHIFT_BINS = np.round(np.arange(TIME_SHIFT_STEP, MAX_TIME_SHIFT + 1e-9, TIME_SHIFT_STEP), 3).tolist()

VELOCITY_BINS = list(range(0, 128, VELOCITY_STEP))

TIME_SHIFT_MAX_INDEX = len(TIME_SHIFT_BINS) - 1



def infer_genre_from_path(midi_path, raw_root):

    try:

        rel = Path(midi_path).relative_to(raw_root)

        if len(rel.parts) >= 2:

            return rel.parts[0].lower()

    except Exception:

        pass

    return "unknown"



def find_midi_files(root):

    exts = {".mid", ".midi"}

    files = sorted([p for p in Path(root).rglob("*") if p.suffix.lower() in exts])

    if MAX_FILES is not None:

        files = files[:MAX_FILES]

    return files



def velocity_to_token(velocity):

    idx = max(0, min(len(VELOCITY_BINS) - 1, int(velocity) // VELOCITY_STEP))

    return f"VELOCITY_{idx}"



def split_time_shift_tokens(delta_seconds):

    tokens = []

    remaining = max(0.0, float(delta_seconds))

    if remaining <= 0:

        return tokens



    while remaining > 1e-9:

        chunk = min(remaining, MAX_TIME_SHIFT)

        idx = int(round(chunk / TIME_SHIFT_STEP)) - 1

        idx = max(0, min(TIME_SHIFT_MAX_INDEX, idx))

        tokens.append(f"TIME_SHIFT_{idx}")

        remaining -= TIME_SHIFT_BINS[idx]

    return tokens



def midi_to_event_tokens(midi_path, skip_drums=True, use_only_piano=False):

    pm = pretty_midi.PrettyMIDI(str(midi_path))

    events = []



    for instrument in pm.instruments:

        if skip_drums and instrument.is_drum:

            continue

        if use_only_piano and instrument.program != 0:

            continue



        for note in instrument.notes:

            start = float(note.start)

            end = float(note.end)

            pitch = int(note.pitch)

            velocity = int(note.velocity)



            # Priority at same timestamp:

            # note_off (0) before velocity (1) before note_on (2)

            events.append((end,   0, f"NOTE_OFF_{pitch}"))

            events.append((start, 1, velocity_to_token(velocity)))

            events.append((start, 2, f"NOTE_ON_{pitch}"))



    events = sorted(events, key=lambda x: (x[0], x[1]))



    tokens = []

    current_time = 0.0

    for event_time, _, token in events:

        delta = event_time - current_time

        tokens.extend(split_time_shift_tokens(delta))

        tokens.append(token)

        current_time = event_time



    return tokens



def process_all_midis(raw_dir):

    midi_files = find_midi_files(raw_dir)

    print("Found MIDI files:", len(midi_files))



    records = []

    bad_files = []



    for path in tqdm(midi_files):

        try:

            tokens = midi_to_event_tokens(

                path,

                skip_drums=SKIP_DRUMS,

                use_only_piano=USE_ONLY_PIANO

            )

            if sum(tok.startswith("NOTE_ON_") for tok in tokens) < MIN_NOTES_PER_FILE:

                continue



            genre = infer_genre_from_path(path, raw_dir)

            records.append({

                "path": str(path),

                "genre": genre,

                "tokens": tokens,

                "n_tokens": len(tokens),

                "n_notes": int(sum(tok.startswith("NOTE_ON_") for tok in tokens))

            })

        except Exception as e:

            bad_files.append({"path": str(path), "error": str(e)})



    return records, bad_files
records, bad_files = process_all_midis(RAW_MIDI_DIR)

print("Usable files:", len(records))

print("Bad files:", len(bad_files))



records_df = pd.DataFrame([{

    "path": r["path"],

    "genre": r["genre"],

    "n_tokens": r["n_tokens"],

    "n_notes": r["n_notes"]

} for r in records])



display(records_df.head())

print(records_df["genre"].value_counts(dropna=False))
if bad_files:

    bad_df = pd.DataFrame(bad_files)

    display(bad_df.head())
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]



def build_token_vocab(records):

    vocab = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}

    for r in records:

        for tok in r["tokens"]:

            if tok not in vocab:

                vocab[tok] = len(vocab)

    return vocab



token_to_id = build_token_vocab(records)

id_to_token = {i: t for t, i in token_to_id.items()}



PAD_IDX = token_to_id["<PAD>"]

BOS_IDX = token_to_id["<BOS>"]

EOS_IDX = token_to_id["<EOS>"]

UNK_IDX = token_to_id["<UNK>"]



all_genres = sorted(set(r["genre"] for r in records))

genre_to_id = {g: i for i, g in enumerate(all_genres)}

id_to_genre = {i: g for g, i in genre_to_id.items()}



print("Vocab size:", len(token_to_id))

print("Genres:", genre_to_id)
def encode_tokens(tokens, token_to_id):

    return [token_to_id.get(tok, UNK_IDX) for tok in tokens]



for r in records:

    r["token_ids"] = encode_tokens(r["tokens"], token_to_id)

    r["genre_id"] = genre_to_id[r["genre"]]
def split_records_by_genre(records, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):

    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6



    grouped = defaultdict(list)

    for r in records:

        grouped[r["genre"]].append(r)



    rng = random.Random(seed)

    train_records, val_records, test_records = [], [], []



    for genre, items in grouped.items():

        items = items[:]

        rng.shuffle(items)

        n = len(items)



        n_train = max(1, int(round(train_ratio * n))) if n >= 3 else max(1, n - 2)

        n_val = int(round(val_ratio * n))

        n_test = n - n_train - n_val



        if n >= 3:

            if n_val == 0:

                n_val = 1

            n_test = max(1, n - n_train - n_val)

            if n_train + n_val + n_test > n:

                n_train = max(1, n - n_val - n_test)

        else:

            n_train = max(1, n - 1)

            n_val = 0

            n_test = n - n_train



        train_records.extend(items[:n_train])

        val_records.extend(items[n_train:n_train+n_val])

        test_records.extend(items[n_train+n_val:n_train+n_val+n_test])



    return train_records, val_records, test_records



train_records, val_records, test_records = split_records_by_genre(

    records,

    train_ratio=TRAIN_RATIO,

    val_ratio=VAL_RATIO,

    test_ratio=TEST_RATIO,

    seed=SEED

)



print("Train files:", len(train_records))

print("Val files:", len(val_records))

print("Test files:", len(test_records))
def summarize_split(name, split_records):

    df = pd.DataFrame([{"genre": r["genre"], "n_tokens": r["n_tokens"]} for r in split_records])

    if len(df) == 0:

        return pd.DataFrame(columns=["genre", "count", "mean_tokens"])

    out = df.groupby("genre").agg(count=("genre", "size"), mean_tokens=("n_tokens", "mean")).reset_index()

    out.insert(0, "split", name)

    return out



split_summary = pd.concat([

    summarize_split("train", train_records),

    summarize_split("val", val_records),

    summarize_split("test", test_records),

], ignore_index=True)



display(split_summary)

split_summary.to_csv(TABLE_DIR / "split_summary.csv", index=False)
def create_fixed_windows(split_records, seq_len=128, stride=64):

    xs, genres, meta = [], [], []

    for r in split_records:

        token_ids = r["token_ids"]

        genre_id = r["genre_id"]

        for start in range(0, len(token_ids) - seq_len + 1, stride):

            window = token_ids[start:start+seq_len]

            xs.append(window)

            genres.append(genre_id)

            meta.append({"path": r["path"], "genre": r["genre"], "start": start})

    if len(xs) == 0:

        return np.zeros((0, seq_len), dtype=np.int64), np.zeros((0,), dtype=np.int64), []

    return np.array(xs, dtype=np.int64), np.array(genres, dtype=np.int64), meta



def create_autoregressive_windows(split_records, seq_len=256, stride=64):

    xs, ys, genres, meta = [], [], [], []

    for r in split_records:

        token_ids = [BOS_IDX] + r["token_ids"] + [EOS_IDX]

        genre_id = r["genre_id"]

        for start in range(0, len(token_ids) - seq_len - 1, stride):

            x = token_ids[start:start+seq_len]

            y = token_ids[start+1:start+seq_len+1]

            xs.append(x)

            ys.append(y)

            genres.append(genre_id)

            meta.append({"path": r["path"], "genre": r["genre"], "start": start})

    if len(xs) == 0:

        return (

            np.zeros((0, seq_len), dtype=np.int64),

            np.zeros((0, seq_len), dtype=np.int64),

            np.zeros((0,), dtype=np.int64),

            []

        )

    return np.array(xs, dtype=np.int64), np.array(ys, dtype=np.int64), np.array(genres, dtype=np.int64), meta



# AE / VAE windows

ae_train_x, ae_train_g, ae_train_meta = create_fixed_windows(train_records, seq_len=SEQ_LEN_AE_VAE, stride=WINDOW_STRIDE)

ae_val_x, ae_val_g, ae_val_meta = create_fixed_windows(val_records, seq_len=SEQ_LEN_AE_VAE, stride=WINDOW_STRIDE)

ae_test_x, ae_test_g, ae_test_meta = create_fixed_windows(test_records, seq_len=SEQ_LEN_AE_VAE, stride=WINDOW_STRIDE)



# Transformer windows

tr_train_x, tr_train_y, tr_train_g, tr_train_meta = create_autoregressive_windows(train_records, seq_len=SEQ_LEN_TRANSFORMER, stride=WINDOW_STRIDE)

tr_val_x, tr_val_y, tr_val_g, tr_val_meta = create_autoregressive_windows(val_records, seq_len=SEQ_LEN_TRANSFORMER, stride=WINDOW_STRIDE)

tr_test_x, tr_test_y, tr_test_g, tr_test_meta = create_autoregressive_windows(test_records, seq_len=SEQ_LEN_TRANSFORMER, stride=WINDOW_STRIDE)



print("AE/VAE train windows:", ae_train_x.shape)

print("Transformer train windows:", tr_train_x.shape)
np.save(PROCESSED_DIR / "ae_train_x.npy", ae_train_x)

np.save(PROCESSED_DIR / "ae_train_g.npy", ae_train_g)

np.save(PROCESSED_DIR / "ae_val_x.npy", ae_val_x)

np.save(PROCESSED_DIR / "ae_val_g.npy", ae_val_g)

np.save(PROCESSED_DIR / "tr_train_x.npy", tr_train_x)

np.save(PROCESSED_DIR / "tr_train_y.npy", tr_train_y)

np.save(PROCESSED_DIR / "tr_train_g.npy", tr_train_g)



with open(PROCESSED_DIR / "token_vocab.json", "w") as f:

    json.dump(token_to_id, f)



with open(PROCESSED_DIR / "genre_vocab.json", "w") as f:

    json.dump(genre_to_id, f)



print("Saved processed arrays and vocab files.")
class WindowDataset(Dataset):

    def __init__(self, xs, genres):

        self.xs = xs

        self.genres = genres



    def __len__(self):

        return len(self.xs)



    def __getitem__(self, idx):

        x = torch.tensor(self.xs[idx], dtype=torch.long)

        g = torch.tensor(self.genres[idx], dtype=torch.long)

        return x, g



class ARDataset(Dataset):

    def __init__(self, xs, ys, genres):

        self.xs = xs

        self.ys = ys

        self.genres = genres



    def __len__(self):

        return len(self.xs)



    def __getitem__(self, idx):

        x = torch.tensor(self.xs[idx], dtype=torch.long)

        y = torch.tensor(self.ys[idx], dtype=torch.long)

        g = torch.tensor(self.genres[idx], dtype=torch.long)

        return x, y, g



def make_loader(dataset, batch_size=BATCH_SIZE, shuffle=True):

    return DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=shuffle,

        num_workers=NUM_WORKERS,

        pin_memory=torch.cuda.is_available(),

        drop_last=False

    )



ae_train_loader = make_loader(WindowDataset(ae_train_x, ae_train_g), shuffle=True)

ae_val_loader = make_loader(WindowDataset(ae_val_x, ae_val_g), shuffle=False) if len(ae_val_x) else None

ae_test_loader = make_loader(WindowDataset(ae_test_x, ae_test_g), shuffle=False) if len(ae_test_x) else None



tr_train_loader = make_loader(ARDataset(tr_train_x, tr_train_y, tr_train_g), shuffle=True)

tr_val_loader = make_loader(ARDataset(tr_val_x, tr_val_y, tr_val_g), shuffle=False) if len(tr_val_x) else None

tr_test_loader = make_loader(ARDataset(tr_test_x, tr_test_y, tr_test_g), shuffle=False) if len(tr_test_x) else None



print("AE/VAE train batches:", len(ae_train_loader))

print("Transformer train batches:", len(tr_train_loader))
def shift_right_for_decoder(targets, bos_idx=BOS_IDX):

    bos = torch.full((targets.size(0), 1), bos_idx, dtype=torch.long, device=targets.device)

    return torch.cat([bos, targets[:, :-1]], dim=1)



def ids_to_tokens(ids):

    tokens = []

    for idx in ids:

        token = id_to_token.get(int(idx), "<UNK>")

        if token not in {"<PAD>", "<BOS>", "<EOS>", "<UNK>"}:

            tokens.append(token)

    return tokens



def token_to_time_shift(token):

    idx = int(token.split("_")[-1])

    idx = max(0, min(idx, len(TIME_SHIFT_BINS) - 1))

    return TIME_SHIFT_BINS[idx]



def token_to_velocity(token):

    idx = int(token.split("_")[-1])

    idx = max(0, min(idx, len(VELOCITY_BINS) - 1))

    return max(1, min(127, VELOCITY_BINS[idx] + VELOCITY_STEP // 2))



def tokens_to_midi(tokens, out_path, program=0):

    pm = pretty_midi.PrettyMIDI()

    instrument = pretty_midi.Instrument(program=program)



    current_time = 0.0

    current_velocity = 80

    active_notes = {}



    for tok in tokens:

        if tok.startswith("TIME_SHIFT_"):

            current_time += token_to_time_shift(tok)

        elif tok.startswith("VELOCITY_"):

            current_velocity = token_to_velocity(tok)

        elif tok.startswith("NOTE_ON_"):

            pitch = int(tok.split("_")[-1])

            active_notes[pitch] = (current_time, current_velocity)

        elif tok.startswith("NOTE_OFF_"):

            pitch = int(tok.split("_")[-1])

            if pitch in active_notes:

                start_time, velocity = active_notes.pop(pitch)

                end_time = max(current_time, start_time + 0.05)

                instrument.notes.append(

                    pretty_midi.Note(

                        velocity=int(velocity),

                        pitch=int(pitch),

                        start=float(start_time),

                        end=float(end_time)

                    )

                )



    pm.instruments.append(instrument)

    pm.write(str(out_path))
def pitch_histogram(tokens):

    pitches = [int(tok.split("_")[-1]) % 12 for tok in tokens if tok.startswith("NOTE_ON_")]

    count = Counter(pitches)

    total = sum(count.values()) if count else 1

    return np.array([count[i] / total for i in range(12)], dtype=np.float32)



def pitch_histogram_distance(tokens_a, tokens_b):

    hist_a = pitch_histogram(tokens_a)

    hist_b = pitch_histogram(tokens_b)

    return float(np.abs(hist_a - hist_b).sum())



def rhythm_diversity(tokens):

    durations = [tok for tok in tokens if tok.startswith("TIME_SHIFT_")]

    if not durations:

        return 0.0

    return float(len(set(durations)) / len(durations))



def repetition_ratio(tokens, n=4):

    patterns = [tuple(tokens[i:i+n]) for i in range(max(0, len(tokens) - n + 1))]

    if not patterns:

        return 0.0

    count = Counter(patterns)

    repeated = sum(v for v in count.values() if v > 1)

    return float(repeated / len(patterns))



def note_density(tokens):

    total_notes = sum(tok.startswith("NOTE_ON_") for tok in tokens)

    total_time = sum(token_to_time_shift(tok) for tok in tokens if tok.startswith("TIME_SHIFT_"))

    if total_time <= 1e-6:

        return 0.0

    return float(total_notes / total_time)



def average_note_duration(tokens):

    durations = [token_to_time_shift(tok) for tok in tokens if tok.startswith("TIME_SHIFT_")]

    if not durations:

        return 0.0

    return float(np.mean(durations))



def summarize_generated_tokens(tokens, ref_tokens=None):

    row = {

        "rhythm_diversity": rhythm_diversity(tokens),

        "repetition_ratio": repetition_ratio(tokens),

        "note_density": note_density(tokens),

        "avg_note_duration": average_note_duration(tokens),

        "n_notes": int(sum(tok.startswith("NOTE_ON_") for tok in tokens))

    }

    if ref_tokens is not None:

        row["pitch_hist_distance_to_ref"] = pitch_histogram_distance(tokens, ref_tokens)

    return row
all_train_token_ids = [tid for r in train_records for tid in r["token_ids"]]



def random_baseline_generate(length=256):

    note_on_tokens = [t for t in token_to_id if t.startswith("NOTE_ON_")]

    note_off_tokens = [t for t in token_to_id if t.startswith("NOTE_OFF_")]

    velocity_tokens = [t for t in token_to_id if t.startswith("VELOCITY_")]

    time_tokens = [t for t in token_to_id if t.startswith("TIME_SHIFT_")]



    tokens = []

    for _ in range(max(1, length // 4)):

        pitch_on = random.choice(note_on_tokens) if note_on_tokens else "NOTE_ON_60"

        pitch = pitch_on.split("_")[-1]

        tokens.append(random.choice(time_tokens) if time_tokens else "TIME_SHIFT_0")

        tokens.append(random.choice(velocity_tokens) if velocity_tokens else "VELOCITY_20")

        tokens.append(pitch_on)

        tokens.append(random.choice(time_tokens) if time_tokens else "TIME_SHIFT_0")

        tokens.append(f"NOTE_OFF_{pitch}")

    return tokens[:length]



def build_markov_chain(train_records):

    transitions = defaultdict(Counter)

    starts = Counter()



    for r in train_records:

        toks = r["tokens"]

        if len(toks) < 2:

            continue

        starts[toks[0]] += 1

        for a, b in zip(toks[:-1], toks[1:]):

            transitions[a][b] += 1

    return starts, transitions



markov_starts, markov_transitions = build_markov_chain(train_records)



def sample_from_counter(counter):

    items = list(counter.items())

    keys = [k for k, _ in items]

    weights = np.array([v for _, v in items], dtype=np.float64)

    weights = weights / weights.sum()

    return np.random.choice(keys, p=weights)



def markov_generate(max_len=256):

    if len(markov_starts) == 0:

        return random_baseline_generate(max_len)

    current = sample_from_counter(markov_starts)

    out = [current]

    for _ in range(max_len - 1):

        next_counter = markov_transitions.get(current)

        if not next_counter:

            current = sample_from_counter(markov_starts)

        else:

            current = sample_from_counter(next_counter)

        out.append(current)

    return out
baseline_rows = []



for name, generator_fn in [("Random", random_baseline_generate), ("Markov", markov_generate)]:

    tokens = generator_fn(length=256)

    out_path = GENERATED_DIR / f"{name.lower()}_baseline_sample.mid"

    tokens_to_midi(tokens, out_path)

    row = {"model": name, **summarize_generated_tokens(tokens)}

    baseline_rows.append(row)



baseline_df = pd.DataFrame(baseline_rows)

display(baseline_df)

baseline_df.to_csv(TABLE_DIR / "baseline_metrics.csv", index=False)
def choose_single_genre(train_records, manual_genre=None):

    genre_counts = Counter(r["genre"] for r in train_records if r["genre"] != "unknown")

    if manual_genre is not None and manual_genre in genre_counts:

        return manual_genre

    if genre_counts:

        return genre_counts.most_common(1)[0][0]

    return Counter(r["genre"] for r in train_records).most_common(1)[0][0]



task1_genre = choose_single_genre(train_records, T1_SINGLE_GENRE)

print("Task 1 selected genre:", task1_genre)
def filter_split_by_genre(xs, genres, selected_genre_id):

    mask = (genres == selected_genre_id)

    return xs[mask], genres[mask]



task1_genre_id = genre_to_id[task1_genre]

t1_train_x, t1_train_g = filter_split_by_genre(ae_train_x, ae_train_g, task1_genre_id)

t1_val_x, t1_val_g = filter_split_by_genre(ae_val_x, ae_val_g, task1_genre_id)

t1_test_x, t1_test_g = filter_split_by_genre(ae_test_x, ae_test_g, task1_genre_id)



t1_train_loader = make_loader(WindowDataset(t1_train_x, t1_train_g), shuffle=True)

t1_val_loader = make_loader(WindowDataset(t1_val_x, t1_val_g), shuffle=False) if len(t1_val_x) else None

t1_test_loader = make_loader(WindowDataset(t1_test_x, t1_test_g), shuffle=False) if len(t1_test_x) else None



print("Task 1 train windows:", len(t1_train_x))

print("Task 1 val windows:", len(t1_val_x))
class LSTMAutoencoder(nn.Module):

    def __init__(self, vocab_size, emb_dim=256, hidden_dim=512, latent_dim=128, num_layers=2):

        super().__init__()

        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)

        self.encoder = nn.LSTM(

            emb_dim,

            hidden_dim,

            num_layers=num_layers,

            batch_first=True,

            bidirectional=True

        )

        self.latent_proj = nn.Linear(hidden_dim * 2, latent_dim)

        self.decoder = nn.LSTM(

            emb_dim + latent_dim,

            hidden_dim,

            num_layers=num_layers,

            batch_first=True

        )

        self.output = nn.Linear(hidden_dim, vocab_size)



    def encode(self, x):

        emb = self.embedding(x)

        _, (h_n, _) = self.encoder(emb)

        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)

        z = self.latent_proj(h)

        return z



    def decode_teacher(self, decoder_input_ids, z):

        emb = self.embedding(decoder_input_ids)

        z_expand = z.unsqueeze(1).repeat(1, decoder_input_ids.size(1), 1)

        dec_in = torch.cat([emb, z_expand], dim=-1)

        out, _ = self.decoder(dec_in)

        logits = self.output(out)

        return logits



    def forward(self, x):

        z = self.encode(x)

        dec_in = shift_right_for_decoder(x)

        logits = self.decode_teacher(dec_in, z)

        return logits, z



    @torch.no_grad()

    def generate_from_latent(self, z, max_len=256, temperature=1.0):

        self.eval()

        generated = []

        current = torch.full((z.size(0),), BOS_IDX, dtype=torch.long, device=z.device)

        hidden = None



        for _ in range(max_len):

            emb = self.embedding(current).unsqueeze(1)

            step_input = torch.cat([emb, z.unsqueeze(1)], dim=-1)

            out, hidden = self.decoder(step_input, hidden)

            logits = self.output(out[:, -1, :]) / temperature

            probs = torch.softmax(logits, dim=-1)

            current = torch.multinomial(probs, num_samples=1).squeeze(1)

            generated.append(current)



        generated = torch.stack(generated, dim=1)

        return generated
def evaluate_autoencoder(model, loader):

    model.eval()

    losses = []



    with torch.no_grad():

        for x, _ in loader:

            x = x.to(device)

            logits, _ = model(x)

            loss = F.cross_entropy(

                logits.reshape(-1, logits.size(-1)),

                x.reshape(-1),

                ignore_index=PAD_IDX

            )

            losses.append(loss.item())



    return float(np.mean(losses)) if losses else None



def train_task1_autoencoder(model, train_loader, val_loader=None, epochs=10, lr=1e-3):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}



    for epoch in range(epochs):

        model.train()

        batch_losses = []



        for x, _ in tqdm(train_loader, desc=f"Task1 Epoch {epoch+1}/{epochs}"):

            x = x.to(device)

            optimizer.zero_grad()

            logits, _ = model(x)

            loss = F.cross_entropy(

                logits.reshape(-1, logits.size(-1)),

                x.reshape(-1),

                ignore_index=PAD_IDX

            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            batch_losses.append(loss.item())



        train_loss = float(np.mean(batch_losses))

        val_loss = evaluate_autoencoder(model, val_loader) if val_loader is not None else None



        history["train_loss"].append(train_loss)

        history["val_loss"].append(val_loss)



        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss}")



    return history
task1_model = LSTMAutoencoder(

    vocab_size=len(token_to_id),

    emb_dim=T1_EMB_DIM,

    hidden_dim=T1_HIDDEN_DIM,

    latent_dim=T1_LATENT_DIM,

    num_layers=T1_NUM_LAYERS

).to(device)



task1_history = train_task1_autoencoder(

    task1_model,

    t1_train_loader,

    t1_val_loader,

    epochs=T1_EPOCHS,

    lr=T1_LR

)
plt.figure(figsize=(8, 5))

plt.plot(task1_history["train_loss"], label="train")

if any(v is not None for v in task1_history["val_loss"]):

    plt.plot(task1_history["val_loss"], label="val")

plt.xlabel("Epoch")

plt.ylabel("Cross-Entropy")

plt.title(f"Task 1 LSTM Autoencoder Loss — {task1_genre}")

plt.legend()

plt.tight_layout()

plt.savefig(PLOT_DIR / "task1_autoencoder_loss.png")

plt.show()
@torch.no_grad()

def generate_task1_samples(model, split_x, n_samples=5, noise_std=0.1, max_len=256, temperature=1.0):

    model.eval()

    samples = []



    chosen = np.random.choice(len(split_x), size=min(n_samples, len(split_x)), replace=False)

    for i, idx in enumerate(chosen):

        x = torch.tensor(split_x[idx:idx+1], dtype=torch.long, device=device)

        z = model.encode(x)

        z = z + noise_std * torch.randn_like(z)

        gen_ids = model.generate_from_latent(z, max_len=max_len, temperature=temperature)[0].cpu().tolist()

        tokens = ids_to_tokens(gen_ids)

        out_path = GENERATED_DIR / f"task1_{task1_genre}_sample_{i+1:02d}.mid"

        tokens_to_midi(tokens, out_path)

        samples.append({"sample_id": f"task1_{i+1:02d}", "path": str(out_path), "tokens": tokens})

    return samples



task1_samples = generate_task1_samples(task1_model, t1_train_x, n_samples=5, max_len=256, temperature=1.0)

pd.DataFrame([{"sample_id": s["sample_id"], "path": s["path"]} for s in task1_samples])
task1_ref_tokens = train_records[0]["tokens"] if train_records else None

task1_metrics = pd.DataFrame([

    {"model": "Task1_Autoencoder", "sample_id": s["sample_id"], **summarize_generated_tokens(s["tokens"], task1_ref_tokens)}

    for s in task1_samples

])

display(task1_metrics)

task1_metrics.to_csv(TABLE_DIR / "task1_metrics.csv", index=False)



torch.save({

    "model_state_dict": task1_model.state_dict(),

    "genre": task1_genre,

    "config": {

        "emb_dim": T1_EMB_DIM,

        "hidden_dim": T1_HIDDEN_DIM,

        "latent_dim": T1_LATENT_DIM,

        "num_layers": T1_NUM_LAYERS

    }

}, PROJECT_ROOT / "task1_autoencoder.pt")
class ConditionalMusicVAE(nn.Module):

    def __init__(self, vocab_size, num_genres, emb_dim=256, hidden_dim=512, latent_dim=128, num_layers=2, genre_emb_dim=32):

        super().__init__()

        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)

        self.genre_embedding = nn.Embedding(num_genres, genre_emb_dim)



        self.encoder = nn.LSTM(

            emb_dim,

            hidden_dim,

            num_layers=num_layers,

            batch_first=True,

            bidirectional=True

        )

        self.mu = nn.Linear(hidden_dim * 2 + genre_emb_dim, latent_dim)

        self.logvar = nn.Linear(hidden_dim * 2 + genre_emb_dim, latent_dim)



        self.decoder = nn.LSTM(

            emb_dim + latent_dim + genre_emb_dim,

            hidden_dim,

            num_layers=num_layers,

            batch_first=True

        )

        self.output = nn.Linear(hidden_dim, vocab_size)



    def encode(self, x, genre_ids):

        emb = self.embedding(x)

        _, (h_n, _) = self.encoder(emb)

        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)

        g = self.genre_embedding(genre_ids)

        h = torch.cat([h, g], dim=-1)

        mu = self.mu(h)

        logvar = self.logvar(h)

        return mu, logvar



    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        return mu + eps * std



    def decode_teacher(self, decoder_input_ids, z, genre_ids):

        emb = self.embedding(decoder_input_ids)

        g = self.genre_embedding(genre_ids)

        z_expand = z.unsqueeze(1).repeat(1, decoder_input_ids.size(1), 1)

        g_expand = g.unsqueeze(1).repeat(1, decoder_input_ids.size(1), 1)

        dec_in = torch.cat([emb, z_expand, g_expand], dim=-1)

        out, _ = self.decoder(dec_in)

        logits = self.output(out)

        return logits



    def forward(self, x, genre_ids):

        mu, logvar = self.encode(x, genre_ids)

        z = self.reparameterize(mu, logvar)

        dec_in = shift_right_for_decoder(x)

        logits = self.decode_teacher(dec_in, z, genre_ids)

        return logits, mu, logvar



    @torch.no_grad()

    def sample(self, genre_ids, max_len=256, temperature=1.0):

        self.eval()

        batch_size = genre_ids.size(0)

        z = torch.randn(batch_size, self.latent_dim, device=genre_ids.device)

        g = self.genre_embedding(genre_ids)

        generated = []

        current = torch.full((batch_size,), BOS_IDX, dtype=torch.long, device=genre_ids.device)

        hidden = None



        for _ in range(max_len):

            emb = self.embedding(current).unsqueeze(1)

            step_input = torch.cat([emb, z.unsqueeze(1), g.unsqueeze(1)], dim=-1)

            out, hidden = self.decoder(step_input, hidden)

            logits = self.output(out[:, -1, :]) / temperature

            probs = torch.softmax(logits, dim=-1)

            current = torch.multinomial(probs, num_samples=1).squeeze(1)

            generated.append(current)



        return torch.stack(generated, dim=1)



    @torch.no_grad()

    def interpolate(self, x_a, g_a, x_b, g_b, steps=5, max_len=256, temperature=1.0):

        mu_a, _ = self.encode(x_a, g_a)

        mu_b, _ = self.encode(x_b, g_b)

        genre_ids = g_a



        all_outputs = []

        for alpha in np.linspace(0, 1, steps):

            z = (1 - alpha) * mu_a + alpha * mu_b

            g = self.genre_embedding(genre_ids)

            generated = []

            current = torch.full((z.size(0),), BOS_IDX, dtype=torch.long, device=z.device)

            hidden = None

            for _ in range(max_len):

                emb = self.embedding(current).unsqueeze(1)

                step_input = torch.cat([emb, z.unsqueeze(1), g.unsqueeze(1)], dim=-1)

                out, hidden = self.decoder(step_input, hidden)

                logits = self.output(out[:, -1, :]) / temperature

                probs = torch.softmax(logits, dim=-1)

                current = torch.multinomial(probs, num_samples=1).squeeze(1)

                generated.append(current)

            all_outputs.append(torch.stack(generated, dim=1))

        return all_outputs
def vae_loss(logits, targets, mu, logvar, beta=0.1):

    recon = F.cross_entropy(

        logits.reshape(-1, logits.size(-1)),

        targets.reshape(-1),

        ignore_index=PAD_IDX

    )

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon + beta * kl, recon, kl



def evaluate_vae(model, loader, beta=0.1):

    model.eval()

    total_loss, total_recon, total_kl = [], [], []



    with torch.no_grad():

        for x, g in loader:

            x = x.to(device)

            g = g.to(device)

            logits, mu, logvar = model(x, g)

            loss, recon, kl = vae_loss(logits, x, mu, logvar, beta=beta)

            total_loss.append(loss.item())

            total_recon.append(recon.item())

            total_kl.append(kl.item())



    return {

        "loss": float(np.mean(total_loss)) if total_loss else None,

        "recon": float(np.mean(total_recon)) if total_recon else None,

        "kl": float(np.mean(total_kl)) if total_kl else None

    }



def train_task2_vae(model, train_loader, val_loader=None, epochs=10, lr=1e-3, beta=0.1):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_recon": [], "train_kl": [], "val_loss": [], "val_recon": [], "val_kl": []}



    for epoch in range(epochs):

        model.train()

        batch_loss, batch_recon, batch_kl = [], [], []



        for x, g in tqdm(train_loader, desc=f"Task2 Epoch {epoch+1}/{epochs}"):

            x = x.to(device)

            g = g.to(device)



            optimizer.zero_grad()

            logits, mu, logvar = model(x, g)

            loss, recon, kl = vae_loss(logits, x, mu, logvar, beta=beta)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()



            batch_loss.append(loss.item())

            batch_recon.append(recon.item())

            batch_kl.append(kl.item())



        history["train_loss"].append(float(np.mean(batch_loss)))

        history["train_recon"].append(float(np.mean(batch_recon)))

        history["train_kl"].append(float(np.mean(batch_kl)))



        if val_loader is not None:

            val_metrics = evaluate_vae(model, val_loader, beta=beta)

            history["val_loss"].append(val_metrics["loss"])

            history["val_recon"].append(val_metrics["recon"])

            history["val_kl"].append(val_metrics["kl"])

        else:

            history["val_loss"].append(None)

            history["val_recon"].append(None)

            history["val_kl"].append(None)



        print(

            f"Epoch {epoch+1}: "

            f"train_loss={history['train_loss'][-1]:.4f}, "

            f"train_recon={history['train_recon'][-1]:.4f}, "

            f"train_kl={history['train_kl'][-1]:.4f}, "

            f"val_loss={history['val_loss'][-1]}"

        )



    return history
task2_model = ConditionalMusicVAE(

    vocab_size=len(token_to_id),

    num_genres=len(genre_to_id),

    emb_dim=T2_EMB_DIM,

    hidden_dim=T2_HIDDEN_DIM,

    latent_dim=T2_LATENT_DIM,

    num_layers=T2_NUM_LAYERS,

    genre_emb_dim=T2_GENRE_EMB_DIM

).to(device)



task2_history = train_task2_vae(

    task2_model,

    ae_train_loader,

    ae_val_loader,

    epochs=T2_EPOCHS,

    lr=T2_LR,

    beta=T2_BETA

)
plt.figure(figsize=(10, 5))

plt.plot(task2_history["train_loss"], label="train_total")

plt.plot(task2_history["train_recon"], label="train_recon")

plt.plot(task2_history["train_kl"], label="train_kl")

if any(v is not None for v in task2_history["val_loss"]):

    plt.plot(task2_history["val_loss"], label="val_total")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Task 2 VAE Training Curves")

plt.legend()

plt.tight_layout()

plt.savefig(PLOT_DIR / "task2_vae_loss.png")

plt.show()
@torch.no_grad()

def generate_task2_samples(model, genre_to_id, n_total=8, max_len=256, temperature=1.0):

    model.eval()

    genre_names = list(genre_to_id.keys())

    samples = []



    for i in range(n_total):

        genre_name = genre_names[i % len(genre_names)]

        genre_id = torch.tensor([genre_to_id[genre_name]], dtype=torch.long, device=device)

        ids = model.sample(genre_id, max_len=max_len, temperature=temperature)[0].cpu().tolist()

        tokens = ids_to_tokens(ids)

        out_path = GENERATED_DIR / f"task2_{genre_name}_sample_{i+1:02d}.mid"

        tokens_to_midi(tokens, out_path)

        samples.append({"sample_id": f"task2_{i+1:02d}", "genre": genre_name, "path": str(out_path), "tokens": tokens})

    return samples



task2_samples = generate_task2_samples(task2_model, genre_to_id, n_total=8, max_len=256, temperature=1.0)

display(pd.DataFrame([{"sample_id": s["sample_id"], "genre": s["genre"], "path": s["path"]} for s in task2_samples]))
task2_metrics = pd.DataFrame([

    {"model": "Task2_VAE", "sample_id": s["sample_id"], "genre": s["genre"], **summarize_generated_tokens(s["tokens"])}

    for s in task2_samples

])

display(task2_metrics)

task2_metrics.to_csv(TABLE_DIR / "task2_metrics.csv", index=False)



torch.save({

    "model_state_dict": task2_model.state_dict(),

    "config": {

        "emb_dim": T2_EMB_DIM,

        "hidden_dim": T2_HIDDEN_DIM,

        "latent_dim": T2_LATENT_DIM,

        "num_layers": T2_NUM_LAYERS,

        "genre_emb_dim": T2_GENRE_EMB_DIM

    }

}, PROJECT_ROOT / "task2_vae.pt")
# Latent interpolation experiment for Task 2

if len(ae_train_x) >= 2:

    idx_a, idx_b = 0, min(1, len(ae_train_x) - 1)

    x_a = torch.tensor(ae_train_x[idx_a:idx_a+1], dtype=torch.long, device=device)

    g_a = torch.tensor(ae_train_g[idx_a:idx_a+1], dtype=torch.long, device=device)

    x_b = torch.tensor(ae_train_x[idx_b:idx_b+1], dtype=torch.long, device=device)

    g_b = torch.tensor(ae_train_g[idx_b:idx_b+1], dtype=torch.long, device=device)



    interpolated = task2_model.interpolate(x_a, g_a, x_b, g_b, steps=5, max_len=256, temperature=1.0)



    interp_rows = []

    for i, ids in enumerate(interpolated):

        ids = ids[0].cpu().tolist()

        tokens = ids_to_tokens(ids)

        out_path = GENERATED_DIR / f"task2_interpolation_{i+1:02d}.mid"

        tokens_to_midi(tokens, out_path)

        interp_rows.append({"step": i, "path": str(out_path)})



    interp_df = pd.DataFrame(interp_rows)

    display(interp_df)

    interp_df.to_csv(TABLE_DIR / "task2_interpolation_files.csv", index=False)
class GenreConditionedTransformer(nn.Module):

    def __init__(

        self,

        vocab_size,

        num_genres,

        emb_dim=256,

        nhead=8,

        num_layers=4,

        ff_dim=1024,

        dropout=0.1,

        max_len=512,

        genre_emb_dim=256

    ):

        super().__init__()

        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)

        self.position_embedding = nn.Embedding(max_len, emb_dim)

        self.genre_embedding = nn.Embedding(num_genres, genre_emb_dim)



        if genre_emb_dim != emb_dim:

            self.genre_proj = nn.Linear(genre_emb_dim, emb_dim)

        else:

            self.genre_proj = nn.Identity()



        encoder_layer = nn.TransformerEncoderLayer(

            d_model=emb_dim,

            nhead=nhead,

            dim_feedforward=ff_dim,

            dropout=dropout,

            batch_first=True

        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output = nn.Linear(emb_dim, vocab_size)



    def forward(self, x, genre_ids):

        batch_size, seq_len = x.shape

        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)



        token_emb = self.token_embedding(x)

        pos_emb = self.position_embedding(pos)

        genre_emb = self.genre_proj(self.genre_embedding(genre_ids)).unsqueeze(1).expand(batch_size, seq_len, -1)



        h = token_emb + pos_emb + genre_emb



        causal_mask = torch.triu(

            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),

            diagonal=1

        )

        h = self.transformer(h, mask=causal_mask)

        logits = self.output(h)

        return logits



    @torch.no_grad()

    def sample(self, genre_ids, max_len=256, temperature=1.0, start_tokens=None):

        self.eval()

        batch_size = genre_ids.size(0)

        if start_tokens is None:

            x = torch.full((batch_size, 1), BOS_IDX, dtype=torch.long, device=genre_ids.device)

        else:

            x = start_tokens



        for _ in range(max_len - x.size(1)):

            logits = self.forward(x, genre_ids)

            next_logits = logits[:, -1, :] / temperature

            probs = torch.softmax(next_logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)

            x = torch.cat([x, next_token], dim=1)

            if x.size(1) >= self.max_len:

                break



        return x
def transformer_loss(logits, targets):

    return F.cross_entropy(

        logits.reshape(-1, logits.size(-1)),

        targets.reshape(-1),

        ignore_index=PAD_IDX

    )



def evaluate_transformer(model, loader):

    model.eval()

    losses = []



    with torch.no_grad():

        for x, y, g in loader:

            x = x.to(device)

            y = y.to(device)

            g = g.to(device)

            logits = model(x, g)

            loss = transformer_loss(logits, y)

            losses.append(loss.item())



    avg_loss = float(np.mean(losses)) if losses else None

    ppl = float(math.exp(avg_loss)) if avg_loss is not None else None

    return {"loss": avg_loss, "perplexity": ppl}



def train_task3_transformer(model, train_loader, val_loader=None, epochs=10, lr=1e-4):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_perplexity": [], "val_loss": [], "val_perplexity": []}



    for epoch in range(epochs):

        model.train()

        losses = []



        for x, y, g in tqdm(train_loader, desc=f"Task3 Epoch {epoch+1}/{epochs}"):

            x = x.to(device)

            y = y.to(device)

            g = g.to(device)



            optimizer.zero_grad()

            logits = model(x, g)

            loss = transformer_loss(logits, y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()



            losses.append(loss.item())



        train_loss = float(np.mean(losses))

        train_ppl = float(math.exp(train_loss))

        history["train_loss"].append(train_loss)

        history["train_perplexity"].append(train_ppl)



        if val_loader is not None:

            val_metrics = evaluate_transformer(model, val_loader)

            history["val_loss"].append(val_metrics["loss"])

            history["val_perplexity"].append(val_metrics["perplexity"])

        else:

            history["val_loss"].append(None)

            history["val_perplexity"].append(None)



        print(

            f"Epoch {epoch+1}: "

            f"train_loss={train_loss:.4f}, train_ppl={train_ppl:.4f}, "

            f"val_loss={history['val_loss'][-1]}, val_ppl={history['val_perplexity'][-1]}"

        )



    return history
task3_model = GenreConditionedTransformer(

    vocab_size=len(token_to_id),

    num_genres=len(genre_to_id),

    emb_dim=T3_EMB_DIM,

    nhead=T3_NHEAD,

    num_layers=T3_NUM_LAYERS,

    ff_dim=T3_FF_DIM,

    dropout=T3_DROPOUT,

    max_len=T3_MAX_GEN_LEN,

    genre_emb_dim=T3_GENRE_EMB_DIM

).to(device)



task3_history = train_task3_transformer(

    task3_model,

    tr_train_loader,

    tr_val_loader,

    epochs=T3_EPOCHS,

    lr=T3_LR

)
plt.figure(figsize=(10, 5))

plt.plot(task3_history["train_loss"], label="train_loss")

if any(v is not None for v in task3_history["val_loss"]):

    plt.plot(task3_history["val_loss"], label="val_loss")

plt.xlabel("Epoch")

plt.ylabel("Cross-Entropy")

plt.title("Task 3 Transformer Loss")

plt.legend()

plt.tight_layout()

plt.savefig(PLOT_DIR / "task3_transformer_loss.png")

plt.show()



plt.figure(figsize=(10, 5))

plt.plot(task3_history["train_perplexity"], label="train_ppl")

if any(v is not None for v in task3_history["val_perplexity"]):

    plt.plot(task3_history["val_perplexity"], label="val_ppl")

plt.xlabel("Epoch")

plt.ylabel("Perplexity")

plt.title("Task 3 Transformer Perplexity")

plt.legend()

plt.tight_layout()

plt.savefig(PLOT_DIR / "task3_transformer_perplexity.png")

plt.show()
@torch.no_grad()

def generate_task3_samples(model, genre_to_id, n_total=10, max_len=512, temperature=1.0):

    model.eval()

    genre_names = list(genre_to_id.keys())

    samples = []



    for i in range(n_total):

        genre_name = genre_names[i % len(genre_names)]

        genre_id = torch.tensor([genre_to_id[genre_name]], dtype=torch.long, device=device)

        ids = model.sample(genre_id, max_len=max_len, temperature=temperature)[0].cpu().tolist()

        tokens = ids_to_tokens(ids)

        out_path = GENERATED_DIR / f"task3_{genre_name}_sample_{i+1:02d}.mid"

        tokens_to_midi(tokens, out_path)

        samples.append({"sample_id": f"task3_{i+1:02d}", "genre": genre_name, "path": str(out_path), "tokens": tokens})

    return samples



task3_samples = generate_task3_samples(task3_model, genre_to_id, n_total=10, max_len=T3_MAX_GEN_LEN, temperature=1.0)

display(pd.DataFrame([{"sample_id": s["sample_id"], "genre": s["genre"], "path": s["path"]} for s in task3_samples]))
task3_metrics = pd.DataFrame([

    {"model": "Task3_Transformer", "sample_id": s["sample_id"], "genre": s["genre"], **summarize_generated_tokens(s["tokens"])}

    for s in task3_samples

])

display(task3_metrics)

task3_metrics.to_csv(TABLE_DIR / "task3_metrics.csv", index=False)



task3_test_eval = evaluate_transformer(task3_model, tr_test_loader) if tr_test_loader is not None else {}

print("Task 3 test metrics:", task3_test_eval)



torch.save({

    "model_state_dict": task3_model.state_dict(),

    "config": {

        "emb_dim": T3_EMB_DIM,

        "nhead": T3_NHEAD,

        "num_layers": T3_NUM_LAYERS,

        "ff_dim": T3_FF_DIM,

        "dropout": T3_DROPOUT,

        "max_len": T3_MAX_GEN_LEN,

        "genre_emb_dim": T3_GENRE_EMB_DIM

    }

}, PROJECT_ROOT / "task3_transformer.pt")
def generate_task4_rating_pack(model, genre_to_id, out_dir, per_genre=3, max_len=256, temperature=1.0):

    out_dir = Path(out_dir)

    midi_dir = out_dir / "candidates"

    midi_dir.mkdir(parents=True, exist_ok=True)



    metadata = []

    for genre_name, genre_id in genre_to_id.items():

        for j in range(per_genre):

            gid = torch.tensor([genre_id], dtype=torch.long, device=device)

            ids = model.sample(gid, max_len=max_len, temperature=temperature)[0].cpu().tolist()

            tokens = ids_to_tokens(ids)

            sample_id = f"{genre_name}_cand_{j+1:02d}"

            midi_path = midi_dir / f"{sample_id}.mid"

            tokens_to_midi(tokens, midi_path)



            metadata.append({

                "sample_id": sample_id,

                "genre": genre_name,

                "midi_path": str(midi_path),

                "token_ids": ids

            })



    metadata_path = out_dir / "candidate_metadata.json"

    with open(metadata_path, "w") as f:

        json.dump(metadata, f, indent=2)



    template_rows = []

    for row in metadata:

        template_rows.append({

            "sample_id": row["sample_id"],

            "genre": row["genre"],

            "participant_id": "",

            "rating": "",

            "comments": ""

        })



    ratings_template = pd.DataFrame(template_rows)

    ratings_template_path = out_dir / "ratings_template.csv"

    ratings_template.to_csv(ratings_template_path, index=False)



    return metadata_path, ratings_template_path



rating_metadata_path, ratings_template_path = generate_task4_rating_pack(

    task3_model,

    genre_to_id,

    RLHF_DIR,

    per_genre=T4_CANDIDATES_PER_GENRE,

    max_len=T4_RL_MAX_LEN,

    temperature=1.0

)



print("Candidate metadata:", rating_metadata_path)

print("Ratings CSV template:", ratings_template_path)
# After collecting ratings, put the completed CSV here:

RATINGS_CSV_PATH = RLHF_DIR / "ratings_filled.csv"



if RATINGS_CSV_PATH.exists():

    ratings_df = pd.read_csv(RATINGS_CSV_PATH)

    display(ratings_df.head())

else:

    print("Please fill ratings_template.csv and save it as ratings_filled.csv in:", RLHF_DIR)
def extract_reward_features(tokens):

    pitch_hist = pitch_histogram(tokens)

    features = np.array([

        rhythm_diversity(tokens),

        repetition_ratio(tokens),

        note_density(tokens),

        average_note_duration(tokens),

        float(sum(tok.startswith("NOTE_ON_") for tok in tokens)),

        float(sum(tok.startswith("TIME_SHIFT_") for tok in tokens)),

        float(np.std(pitch_hist)),

        float(np.max(pitch_hist) if len(pitch_hist) else 0.0),

    ], dtype=np.float32)

    return features



class RewardDataset(Dataset):

    def __init__(self, x_features, ratings):

        self.x = torch.tensor(x_features, dtype=torch.float32)

        self.y = torch.tensor(ratings, dtype=torch.float32).unsqueeze(1)



    def __len__(self):

        return len(self.x)



    def __getitem__(self, idx):

        return self.x[idx], self.y[idx]



class RewardModel(nn.Module):

    def __init__(self, input_dim, hidden_dim=64):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),

            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),

            nn.ReLU(),

            nn.Linear(hidden_dim, 1)

        )



    def forward(self, x):

        return self.net(x)
def load_rated_candidates(metadata_json_path, ratings_csv_path):

    with open(metadata_json_path, "r") as f:

        metadata = json.load(f)

    ratings_df = pd.read_csv(ratings_csv_path)



    agg = ratings_df.groupby("sample_id", as_index=False)["rating"].mean()

    meta_df = pd.DataFrame(metadata)

    merged = meta_df.merge(agg, on="sample_id", how="inner")

    merged["tokens"] = merged["token_ids"].apply(ids_to_tokens)

    merged["genre_id"] = merged["genre"].map(genre_to_id)

    return merged



if RATINGS_CSV_PATH.exists():

    rated_df = load_rated_candidates(rating_metadata_path, RATINGS_CSV_PATH)

    rated_df["feature_vec"] = rated_df["tokens"].apply(extract_reward_features)

    display(rated_df[["sample_id", "genre", "rating"]].head())

else:

    rated_df = None
def train_reward_model(rated_df, epochs=50, lr=1e-3, hidden_dim=64):

    X = np.stack(rated_df["feature_vec"].values)

    y = rated_df["rating"].values.astype(np.float32)



    dataset = RewardDataset(X, y)

    loader = make_loader(dataset, batch_size=min(16, len(dataset)), shuffle=True)



    model = RewardModel(input_dim=X.shape[1], hidden_dim=hidden_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []



    for epoch in range(epochs):

        model.train()

        losses = []



        for x_batch, y_batch in loader:

            x_batch = x_batch.to(device)

            y_batch = y_batch.to(device)



            optimizer.zero_grad()

            pred = model(x_batch)

            loss = F.mse_loss(pred, y_batch)

            loss.backward()

            optimizer.step()

            losses.append(loss.item())



        avg_loss = float(np.mean(losses))

        history.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:

            print(f"Reward epoch {epoch+1}: mse={avg_loss:.4f}")



    return model, history



if rated_df is not None and len(rated_df) >= 4:

    reward_model, reward_history = train_reward_model(

        rated_df,

        epochs=T4_REWARD_EPOCHS,

        lr=T4_REWARD_LR,

        hidden_dim=T4_REWARD_HIDDEN

    )

else:

    reward_model, reward_history = None, None

    print("Need at least a few rated samples to train the reward model.")
if reward_history is not None:

    plt.figure(figsize=(8, 5))

    plt.plot(reward_history)

    plt.xlabel("Epoch")

    plt.ylabel("MSE")

    plt.title("Task 4 Reward Model Training Loss")

    plt.tight_layout()

    plt.savefig(PLOT_DIR / "task4_reward_model_loss.png")

    plt.show()
def compute_sequence_logprob(policy_model, seq_ids, genre_ids):

    # seq_ids includes BOS + generated tokens

    x = seq_ids[:, :-1]

    y = seq_ids[:, 1:]

    logits = policy_model(x, genre_ids)

    log_probs = F.log_softmax(logits, dim=-1)

    token_log_probs = log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)

    seq_logprob = token_log_probs.sum(dim=1)

    return seq_logprob



@torch.no_grad()

def sample_policy_batch(policy_model, batch_size, max_len, genre_choices):

    sampled_ids = []

    sampled_genres = []

    sampled_tokens = []



    for _ in range(batch_size):

        genre_name = random.choice(genre_choices)

        genre_id = genre_to_id[genre_name]

        gid = torch.tensor([genre_id], dtype=torch.long, device=device)

        seq = policy_model.sample(gid, max_len=max_len, temperature=1.0)

        ids = seq[0].cpu().tolist()

        tokens = ids_to_tokens(ids)

        sampled_ids.append(ids)

        sampled_genres.append(genre_id)

        sampled_tokens.append(tokens)



    max_seq = max(len(s) for s in sampled_ids)

    padded = []

    for seq in sampled_ids:

        seq = seq + [EOS_IDX]

        if len(seq) < max_seq:

            seq = seq + [PAD_IDX] * (max_seq - len(seq))

        padded.append(seq)



    return (

        torch.tensor(padded, dtype=torch.long, device=device),

        torch.tensor(sampled_genres, dtype=torch.long, device=device),

        sampled_tokens

    )



def predict_rewards(reward_model, sampled_tokens):

    feats = np.stack([extract_reward_features(toks) for toks in sampled_tokens]).astype(np.float32)

    x = torch.tensor(feats, dtype=torch.float32, device=device)

    with torch.no_grad():

        pred = reward_model(x).squeeze(1)

    return pred
def rlhf_fine_tune(policy_model, reward_model, steps=50, rl_lr=5e-5, batch_size=8, max_len=256):

    policy_model = copy.deepcopy(policy_model).to(device)

    optimizer = torch.optim.Adam(policy_model.parameters(), lr=rl_lr)

    history = []

    genre_choices = list(genre_to_id.keys())



    for step in range(steps):

        seq_ids, genre_ids, sampled_tokens = sample_policy_batch(

            policy_model,

            batch_size=batch_size,

            max_len=max_len,

            genre_choices=genre_choices

        )



        rewards = predict_rewards(reward_model, sampled_tokens)

        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)



        optimizer.zero_grad()

        seq_logprob = compute_sequence_logprob(policy_model, seq_ids, genre_ids)

        loss = -(rewards * seq_logprob).mean()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)

        optimizer.step()



        history.append({

            "step": step + 1,

            "loss": float(loss.item()),

            "mean_reward": float(rewards.mean().item()),

            "std_reward": float(rewards.std().item())

        })



        if (step + 1) % 10 == 0 or step == 0:

            print(

                f"RL step {step+1}: loss={history[-1]['loss']:.4f}, "

                f"mean_reward={history[-1]['mean_reward']:.4f}, "

                f"std_reward={history[-1]['std_reward']:.4f}"

            )



    return policy_model, pd.DataFrame(history)



if reward_model is not None:

    task4_model, task4_history = rlhf_fine_tune(

        task3_model,

        reward_model,

        steps=T4_RL_STEPS,

        rl_lr=T4_RL_LR,

        batch_size=T4_RL_BATCH_SIZE,

        max_len=T4_RL_MAX_LEN

    )

else:

    task4_model, task4_history = None, None

    print("Skipping RLHF fine-tuning until reward_model is available.")
if task4_history is not None:

    display(task4_history.head())

    task4_history.to_csv(TABLE_DIR / "task4_rlhf_history.csv", index=False)



    plt.figure(figsize=(8, 5))

    plt.plot(task4_history["step"], task4_history["loss"])

    plt.xlabel("RL Step")

    plt.ylabel("Policy Gradient Loss")

    plt.title("Task 4 RLHF Fine-Tuning")

    plt.tight_layout()

    plt.savefig(PLOT_DIR / "task4_rlhf_loss.png")

    plt.show()
@torch.no_grad()

def generate_task4_samples(model, genre_to_id, n_total=10, max_len=256, temperature=1.0):

    if model is None:

        return []

    genre_names = list(genre_to_id.keys())

    samples = []



    for i in range(n_total):

        genre_name = genre_names[i % len(genre_names)]

        genre_id = torch.tensor([genre_to_id[genre_name]], dtype=torch.long, device=device)

        ids = model.sample(genre_id, max_len=max_len, temperature=temperature)[0].cpu().tolist()

        tokens = ids_to_tokens(ids)

        out_path = GENERATED_DIR / f"task4_{genre_name}_sample_{i+1:02d}.mid"

        tokens_to_midi(tokens, out_path)

        samples.append({"sample_id": f"task4_{i+1:02d}", "genre": genre_name, "path": str(out_path), "tokens": tokens})

    return samples



task4_samples = generate_task4_samples(task4_model, genre_to_id, n_total=10, max_len=T4_RL_MAX_LEN, temperature=1.0)

if task4_samples:

    display(pd.DataFrame([{"sample_id": s["sample_id"], "genre": s["genre"], "path": s["path"]} for s in task4_samples]))

else:

    print("No Task 4 samples yet. Complete ratings + reward model training first.")
if task4_samples:

    task4_metrics = pd.DataFrame([

        {"model": "Task4_RLHF", "sample_id": s["sample_id"], "genre": s["genre"], **summarize_generated_tokens(s["tokens"])}

        for s in task4_samples

    ])

    display(task4_metrics)

    task4_metrics.to_csv(TABLE_DIR / "task4_metrics.csv", index=False)



    torch.save({

        "model_state_dict": task4_model.state_dict(),

        "config": {

            "emb_dim": T3_EMB_DIM,

            "nhead": T3_NHEAD,

            "num_layers": T3_NUM_LAYERS,

            "ff_dim": T3_FF_DIM,

            "dropout": T3_DROPOUT,

            "max_len": T3_MAX_GEN_LEN,

            "genre_emb_dim": T3_GENRE_EMB_DIM

        }

    }, PROJECT_ROOT / "task4_rlhf_transformer.pt")
def aggregate_metric_table():

    tables = []



    baseline_path = TABLE_DIR / "baseline_metrics.csv"

    t1_path = TABLE_DIR / "task1_metrics.csv"

    t2_path = TABLE_DIR / "task2_metrics.csv"

    t3_path = TABLE_DIR / "task3_metrics.csv"

    t4_path = TABLE_DIR / "task4_metrics.csv"



    if baseline_path.exists():

        tables.append(pd.read_csv(baseline_path))

    if t1_path.exists():

        tables.append(pd.read_csv(t1_path))

    if t2_path.exists():

        tables.append(pd.read_csv(t2_path))

    if t3_path.exists():

        tables.append(pd.read_csv(t3_path))

    if t4_path.exists():

        tables.append(pd.read_csv(t4_path))



    if not tables:

        return None



    df = pd.concat(tables, ignore_index=True)



    agg = df.groupby("model").agg(

        rhythm_diversity=("rhythm_diversity", "mean"),

        repetition_ratio=("repetition_ratio", "mean"),

        note_density=("note_density", "mean"),

        avg_note_duration=("avg_note_duration", "mean"),

        n_notes=("n_notes", "mean"),

    ).reset_index()



    return agg



comparison_df = aggregate_metric_table()

if comparison_df is not None:

    if "Task3_Transformer" in comparison_df["model"].values and tr_test_loader is not None:

        ppl = evaluate_transformer(task3_model, tr_test_loader)["perplexity"]

        comparison_df["perplexity"] = np.nan

        comparison_df.loc[comparison_df["model"] == "Task3_Transformer", "perplexity"] = ppl



    display(comparison_df)

    comparison_df.to_csv(TABLE_DIR / "final_comparison_table.csv", index=False)

else:

    print("Run the model sections first to build the final comparison table.")
