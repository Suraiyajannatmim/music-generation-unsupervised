"""Symbolic MIDI tokenization utilities."""
SPECIAL_TOKENS = ["PAD", "BOS", "EOS", "UNK"]


def velocity_to_token(velocity):
    bucket = int(max(0, min(127, velocity)) // 16)
    return f"VELOCITY_{bucket}"


def build_token_vocab(records):
    vocab = list(SPECIAL_TOKENS)
    seen = set(vocab)
    for record in records:
        for token in record.get("tokens", []):
            if token not in seen:
                seen.add(token)
                vocab.append(token)
    token_to_id = {tok: idx for idx, tok in enumerate(vocab)}
    id_to_token = {idx: tok for tok, idx in token_to_id.items()}
    return token_to_id, id_to_token


def encode_tokens(tokens, token_to_id):
    unk = token_to_id.get("UNK", 3)
    return [token_to_id.get(tok, unk) for tok in tokens]
