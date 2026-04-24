from collections import Counter


def pitch_histogram(tokens):
    counts = Counter()
    for t in tokens:
        if isinstance(t, str) and t.startswith("NOTE_ON_"):
            counts[int(t.split("_")[-1]) % 12] += 1
    total = sum(counts.values()) or 1
    return [counts[i] / total for i in range(12)]


def pitch_histogram_distance(tokens_a, tokens_b):
    a, b = pitch_histogram(tokens_a), pitch_histogram(tokens_b)
    return sum(abs(x - y) for x, y in zip(a, b))


def rhythm_diversity(tokens):
    durations = [t for t in tokens if isinstance(t, str) and t.startswith("TIME_SHIFT_")]
    return len(set(durations)) / max(1, len(durations))


def repetition_ratio(tokens, n=4):
    patterns = [tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1))]
    return 1 - (len(set(patterns)) / max(1, len(patterns)))
