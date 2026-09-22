#!/usr/bin/env python3
"""Configuration for every HerVoice-BN service, read ONLY from the environment.

Docker-readiness without Docker: nothing here reads a file, resolves a path relative to the
current working directory, or assumes a sibling process. Each service is startable from any
cwd with a handful of env vars, so a Dockerfile later is a packaging exercise rather than a
rewrite. The defaults are what the shared box needs today.

Ports: a co-tenant on this box owns 8080, 8081, 8082, 8889, 9000-9005 and 9011-9015. These
defaults deliberately avoid all of them. Everything binds to 127.0.0.1; the browser reaches
the gateway over an SSH tunnel, which also satisfies getUserMedia's secure-context rule
because http://localhost counts as secure.
"""
import os


def _s(name, default):
    return os.environ.get(name, default)


def _i(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _f(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _b(name, default=False):
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------- addresses
ASR_HOST = _s("HV_ASR_HOST", "127.0.0.1")
ASR_PORT = _i("HV_ASR_PORT", 8001)
TTS_HOST = _s("HV_TTS_HOST", "127.0.0.1")
TTS_PORT = _i("HV_TTS_PORT", 8002)
GW_HOST = _s("HV_GW_HOST", "127.0.0.1")
GW_PORT = _i("HV_GW_PORT", 8100)

# Where the gateway finds the others. In Docker these become service names; nothing else
# about the code changes.
ASR_URL = _s("HV_ASR_URL", f"http://{ASR_HOST}:{ASR_PORT}")
TTS_URL = _s("HV_TTS_URL", f"http://{TTS_HOST}:{TTS_PORT}")
LLM_URL = _s("HV_LLM_URL", "http://127.0.0.1:8090")
LLM_MODEL = _s("HV_LLM_MODEL", "gemma4-e4b")

# ------------------------------------------------------------------------------- models
ASR_MODEL = _s("HV_ASR_MODEL", "ehzawad/stt_bn_fastconformer_ctc")
TTS_REPO = _s("HV_TTS_REPO", "ehzawad/indicf5-bangla-tts")
TTS_REF_REPO = _s("HV_TTS_REF_REPO", "ai4bharat/IndicF5")
TTS_REF_FILE = _s("HV_TTS_REF_FILE", "prompts/PAN_F_HAPPY_00001.wav")
# Override both together to use a different assistant voice.
TTS_REF_WAV = _s("HV_TTS_REF_WAV", "")
TTS_REF_TEXT = _s("HV_TTS_REF_TEXT", "")

DEVICE = _s("HV_DEVICE", "cuda")

# --------------------------------------------------------------------------- generation
# NFE 16 measured ~1553 ms per chunk against ~3074 ms at 32 for the same text, and is the
# single biggest latency lever in the whole pipeline. Its quality cost is NOT yet measured,
# so it stays configurable and the default is stated explicitly rather than inherited.
TTS_NFE = _i("HV_TTS_NFE", 16)
TTS_CFG = _f("HV_TTS_CFG", 2.0)
TTS_SWAY = _f("HV_TTS_SWAY", -1.0)
TTS_SPEED = _f("HV_TTS_SPEED", 1.0)
TTS_MAX_BYTES = _i("HV_TTS_MAX_BYTES", 400)

LLM_MAX_TOKENS = _i("HV_LLM_MAX_TOKENS", 160)

# Conversation memory (text only -- see conversation.py). Six exchanges and ~2400 chars keep
# well inside Gemma's 2048-token window even at 1 token per 2 Bengali characters.
MEM_MAX_TURNS = _i("HV_MEM_MAX_TURNS", 12)
MEM_MAX_CHARS = _i("HV_MEM_MAX_CHARS", 2400)
LLM_TEMPERATURE = _f("HV_LLM_TEMPERATURE", 0.0)

SYSTEM_PROMPT = _s(
    "HV_SYSTEM_PROMPT",
    "তুমি একজন সহায়ক বাংলা কণ্ঠ-সহকারী। সবসময় বাংলায় উত্তর দাও। "
    "উত্তর সংক্ষিপ্ত রাখো — দুই থেকে তিনটি ছোট বাক্য।",
)

# ------------------------------------------------------------------------ turn taking
# End-of-turn silence. Measured on 30 real spontaneous Bengali clips (IndicVoices-R extempore):
# clips cut off mid-sentence / median added latency -- 220 ms: 15/30 / 242 ms; 350: 9 / 368;
# 500: 6 / 528; 600: 3 / 624; 700: 1 / 722; 800: 1 / 817; 1000: 0 / 1042. Silero's default 220
# fired at natural mid-sentence pauses and truncated 7.8 s of speech to one word. 600 is the
# knee: an 80 % cut in interruptions for +380 ms. A semantic end-of-turn model (Smart Turn v3,
# published Bengali 83.8 %) was measured on the same clips and did NOT help: 15 -> 13 cut off,
# 5/30 real ends missed, 101 ms per decision. Plain silence wins here. 700 over 600: another
# 98 ms took cut-off clips from 3 to 1; a cut-off destroys the referent AND turns the
# continuation into a barge-in, so the exchange is worth it. Unproven for short questions.
MIN_SILENCE_MS = _i("HV_MIN_SILENCE_MS", 700)
MIN_SPEECH_MS = _i("HV_MIN_SPEECH_MS", 120)

# ------------------------------------------------------------------------------- audio
SR_IN = 16000      # everything upstream of the brain
SR_OUT = 24000     # IndicF5 / Vocos output

# ------------------------------------------------------------------------------ limits
# The gateway must never let a slow consumer turn into unbounded memory: the live loop's
# own frame queue is unbounded upstream, so admission is bounded here instead.
MAX_INBOUND_FRAMES = _i("HV_MAX_INBOUND_FRAMES", 200)      # ~4 s at 20 ms frames
MAX_TURN_SECONDS = _f("HV_MAX_TURN_SECONDS", 30.0)
HTTP_TIMEOUT_S = _f("HV_HTTP_TIMEOUT_S", 60.0)
READY_TIMEOUT_S = _f("HV_READY_TIMEOUT_S", 600.0)

# A shared token, required even on localhost: an SSH tunnel is reachable by anyone else with
# an account on this box, and this endpoint accepts live microphone audio.
GW_TOKEN = _s("HV_GW_TOKEN", "")
MAX_SESSIONS = _i("HV_MAX_SESSIONS", 1)   # one GPU budget, one conversation


def summary():
    return {
        "asr_url": ASR_URL, "tts_url": TTS_URL, "llm_url": LLM_URL, "llm_model": LLM_MODEL,
        "asr_model": ASR_MODEL, "tts_repo": TTS_REPO, "device": DEVICE,
        "tts_nfe": TTS_NFE, "sr_in": SR_IN, "sr_out": SR_OUT,
        "max_sessions": MAX_SESSIONS, "auth": bool(GW_TOKEN),
        "min_silence_ms": MIN_SILENCE_MS, "mem_turns": MEM_MAX_TURNS,
    }


# MEASURED NULL RESULT, recorded so it is not re-attempted: prompting for a short FIRST
# sentence looked like a 480 ms saving on the critical path, but the saving came entirely from
# the model emitting contentless filler ("প্রয়োজনীয় কাগজপত্র তালিকা নিচে দেওয়া হলো।" -- and
# "below" is meaningless in speech). Adding an explicit ban on filler and preambles collapsed
# the saving to 48 ms, because Gemma already answers directly and its first sentences are
# already as short as the content allows. Do not add first-sentence-length instructions to
# SYSTEM_PROMPT: they trade answer quality for a metric.
