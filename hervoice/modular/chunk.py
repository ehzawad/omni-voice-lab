#!/usr/bin/env python3
"""Small, robust sentence splitter for sentence-chunked streaming TTS.

split_sentences(text) -> list[str]

Splits a brain answer into sentence-ish chunks on `.?!` boundaries (and hard newlines) so the TTS
worker can synthesize and emit the FIRST short sentence's audio as soon as it is ready, instead of
waiting for the whole answer to finish autoregressive generation. That cuts *perceived* latency
(time-to-first-audio), not total generation time.

Design goals (deliberately minimal -- do NOT over-engineer):
  * Split on sentence terminators `.?!` (runs like "?!" count once) followed by whitespace/end.
  * Split on explicit newlines (hard boundaries).
  * Keep list-y content in ONE chunk: "1958, 1962, 1970, 1994, and 2002." has commas, not
    terminators, so it never splits mid-list.
  * Do NOT split inside decimals ("3.14") or after a minimal set of abbreviations ("Dr.", "e.g.")
    or single-letter initials ("J.").
  * Merge stray tiny fragments (no alphanumerics, e.g. a lone "?") into the previous chunk so we
    never emit an empty/degenerate "sentence".

Stdlib only.
"""
import re

# Minimal abbreviation set: enough to avoid the common false splits, not a linguistics project.
_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "no", "fig", "al",
    "inc", "ltd", "co", "e.g", "i.e", "u.s", "u.k", "a.m", "p.m", "gen", "gov", "sen",
}

# A sentence terminator run, optional trailing closing quotes/brackets, then whitespace or end.
_BOUNDARY = re.compile(r"([.!?]+)([)\]\"'’”]*)(\s+|$)")


def _has_alnum(s: str) -> bool:
    return any(c.isalnum() for c in s)


def _split_line(line: str):
    """Sentence-split a single line (no newlines) into a list of trimmed chunks."""
    out = []
    start = 0
    n = len(line)
    for m in _BOUNDARY.finditer(line):
        punct_start = m.start(1)
        first_punct = line[punct_start]
        end = m.end(2)  # just past the punctuation (+ closing quotes), before the whitespace

        # Decimal guard: a '.' with a digit right before AND a digit as the next non-space char
        # is a decimal point (e.g. "3.14"), not a sentence end.
        if first_punct == "." and punct_start > 0 and line[punct_start - 1].isdigit():
            nxt = m.end()
            if nxt < n and line[nxt].isdigit():
                continue

        # Abbreviation / initial guard: look at the last whitespace-delimited token before the dot.
        if first_punct == ".":
            prev = line[start:punct_start]
            last_word = re.split(r"\s", prev)[-1].strip(".").lower()
            if last_word in _ABBREV:
                continue
            if len(last_word) == 1 and last_word.isalpha():  # single-letter initial "J."
                continue

        seg = line[start:end].strip()
        if seg:
            out.append(seg)
        start = m.end()

    tail = line[start:].strip()
    if tail:
        out.append(tail)
    if not out:
        out = [line.strip()]
    return out


def split_sentences(text: str):
    """Split `text` into sentence-ish chunks. Returns [] for empty/whitespace-only input."""
    text = (text or "").strip()
    if not text:
        return []

    raw = []
    for line in re.split(r"[\r\n]+", text):
        line = line.strip()
        if line:
            raw.extend(_split_line(line))

    # Merge fragments with no alphanumerics (a stray "?", "..." etc.) into the previous chunk,
    # so every emitted chunk is something the TTS can actually voice.
    merged = []
    for seg in raw:
        if merged and not _has_alnum(seg):
            merged[-1] = (merged[-1] + " " + seg).strip()
        else:
            merged.append(seg)
    return merged


if __name__ == "__main__":
    import sys
    tests = [
        "Brazil has won the men's World Cup five times, in the years 1958, 1962, 1970, 1994, and 2002.",
        "Brazil won five times. The years were 1958, 1962, 1970, 1994, and 2002.",
        "The capital of France is Paris. It has about 2.1 million residents, and it is on the Seine.",
        "Dr. Smith arrived at 3 p.m. He was late. Why? Nobody knew.",
        "Line one\nLine two.\nLine three!",
        "",
        "   ",
    ]
    args = sys.argv[1:]
    for t in (args if args else tests):
        print(repr(t), "->", split_sentences(t))
