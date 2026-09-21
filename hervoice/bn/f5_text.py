"""Bengali text utilities for the F5 path (shared by data prep, training, inference, evaluation).
- normalize(): the SAME normalisation as prepare_bntts.norm_text (NFC, |→।, strip stray chars, collapse spaces)
- chunk_bn(): danda-aware chunking under a UTF-8 BYTE budget (upstream chunk_text ignores '।' and cannot split an
  oversized sentence); falls back to whitespace boundaries, never splits inside a word/combining sequence
- est_duration_frames(): upstream's duration rule (byte-length ratio), WITHOUT the hidden speed=0.3 for <10-byte inputs
- vocab_missing(): characters of a text not in the checkpoint vocabulary (space excluded; F5 maps unknown -> idx 0 = space)"""
import re, unicodedata

def normalize(t):
    t = unicodedata.normalize("NFC", t).replace("﻿", "").replace("​", "")
    t = t.replace("|", "।"); t = re.sub(r"[•=]", " ", t); t = re.sub(r"\s*/\s*", " ", t)
    t = re.sub(r"\s+([।,.!?;:])", r"\1", t); return re.sub(r"\s+", " ", t).strip()

_TERM = re.compile(r"(?<=[।!?])\s*")          # sentence terminators (danda first-class)
_CLAUSE = re.compile(r"(?<=[;:,.])\s+")         # clause boundaries, used only for oversized sentences
def _blen(s): return len(s.encode("utf-8"))
def _pack(units, max_bytes):
    out, cur = [], ""
    for u in units:
        cand = (cur + " " + u).strip() if cur else u
        if _blen(cand) <= max_bytes or not cur: cur = cand
        else: out.append(cur); cur = u
    if cur: out.append(cur)
    return out
def chunk_bn(text, max_bytes=400):
    """Sentence-first chunking under a UTF-8 byte budget (400 B ~ 133 Bengali chars ~ 10 s of speech).
    Whole sentences are packed greedily; a sentence over budget is split at clause boundaries, then at spaces.
    Never splits inside a word or combining sequence."""
    sents = [u.strip() for u in _TERM.split(text) if u and u.strip()]
    units = []
    for s_ in sents:
        if _blen(s_) <= max_bytes: units.append(s_); continue
        for c in [c.strip() for c in _CLAUSE.split(s_) if c.strip()]:
            units += [c] if _blen(c) <= max_bytes else _pack(c.split(" "), max_bytes)
    return _pack(units, max_bytes)

def est_duration_frames(ref_audio_frames, ref_text, gen_text, speed=1.0):
    """total mel frames = ref frames + ref_frames * gen_bytes / ref_bytes / speed (upstream rule, sans <10-byte hack)."""
    rb, gb = len(ref_text.encode("utf-8")), len(gen_text.encode("utf-8"))
    return int(ref_audio_frames + int(ref_audio_frames / max(rb, 1) * gb / speed))

def load_vocab(path):
    vocab = [l.rstrip("\n") for l in open(path, encoding="utf-8")]
    return vocab, {c: i for i, c in enumerate(vocab)}

def vocab_missing(text, vocab_set):
    return {c for c in text if c != " " and c not in vocab_set}
