#!/usr/bin/env python3
"""Conversation memory for a CASCADED voice bot.

Text is the right thing to remember, and this is a decision, not an omission. In an
ASR -> LLM -> TTS cascade the brain only ever sees text; the audio of earlier turns carries
nothing the model could use, because the model has no audio input. Audio context matters only
for end-to-end speech LLMs (Moshi, GPT-4o realtime, Gemini Live) where the network attends over
past audio tokens natively. Keeping audio here would cost memory and buy nothing.

What IS kept per turn is the ASR transcript as the user turn and the assistant's full reply as
the assistant turn -- plus, for diagnostics, the ASR confidence proxy and timing.

Budgeting is two-tier so a pathological turn cannot blow the context:
  * a hard cap on the number of turns kept (older ones roll off),
  * a soft cap on total characters, which is a conservative stand-in for tokens for Bengali
    (Gemma's tokenizer spends roughly 1 token per 2-3 Bengali characters; we budget as if it
    were 1 per 2 so we never overshoot the model's 2048-token window).

Summarisation of dropped turns is deliberately NOT done in v1: it adds an LLM call to the
critical path and its failure modes (hallucinated summaries) are worse than forgetting.
"""
import threading
import time
from dataclasses import dataclass, field


@dataclass
class Turn:
    role: str            # "user" | "assistant"
    text: str
    t: float = field(default_factory=time.time)
    meta: dict = field(default_factory=dict)


class Conversation:
    def __init__(self, system: str, max_turns: int = 12, max_chars: int = 2400):
        """
        max_turns: user+assistant messages kept (12 = six exchanges).
        max_chars: total characters of history sent, excluding the system prompt.
        """
        self.system = system
        self.max_turns = max_turns
        self.max_chars = max_chars
        self._turns: list[Turn] = []
        self._lock = threading.Lock()
        self.started = time.time()

    # ------------------------------------------------------------------ mutate
    def add_user(self, text: str, **meta):
        self._add(Turn("user", text.strip(), meta=meta))

    def add_assistant(self, text: str, **meta):
        self._add(Turn("assistant", text.strip(), meta=meta))

    def _add(self, turn: Turn):
        if not turn.text:
            return
        with self._lock:
            self._turns.append(turn)
            self._trim()

    def _trim(self):
        # 1. turn-count cap
        while len(self._turns) > self.max_turns:
            self._turns.pop(0)
        # 2. character cap: drop oldest PAIRS so we never leave a dangling assistant turn
        while sum(len(t.text) for t in self._turns) > self.max_chars and len(self._turns) > 2:
            self._turns.pop(0)
            if self._turns and self._turns[0].role == "assistant":
                self._turns.pop(0)

    def drop_last_assistant_if_cancelled(self):
        """A barged-in answer was never fully heard; remembering it as if it were would make
        the next reply refer to things the user did not get. Remove it."""
        with self._lock:
            if self._turns and self._turns[-1].role == "assistant":
                self._turns.pop()

    def reset(self):
        with self._lock:
            self._turns.clear()

    # -------------------------------------------------------------------- read
    def messages(self, pending_user: str | None = None) -> list[dict]:
        """OpenAI-style messages: system + history (+ the in-flight user turn)."""
        with self._lock:
            msgs = [{"role": "system", "content": self.system}]
            msgs += [{"role": t.role, "content": t.text} for t in self._turns]
        if pending_user:
            msgs.append({"role": "user", "content": pending_user.strip()})
        return msgs

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [{"role": t.role, "text": t.text, "t": t.t, **t.meta} for t in self._turns]

    def __len__(self):
        return len(self._turns)
