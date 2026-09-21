#!/usr/bin/env python3
"""Wire protocol between the browser and the gateway.

Two channels on one WebSocket:

  * TEXT frames  -- JSON control and events, both directions.
  * BINARY frames -- PCM. Client to server is bare little-endian float32 mono at 16 kHz.
    Server to client carries a 16-byte header so the client can DROP STALE AUDIO.

Why the header exists. The assistant's audio is produced a whole sentence at a time and sits
in the browser's playback buffer, so at any moment the client may hold seconds of speech that
the server has already decided to abandon. Cancellation therefore cannot mean "stop sending";
it must mean "everything you still hold for epoch N is void". Every audio frame is stamped
with the epoch it belongs to, and the client discards any frame whose epoch is not current.

    offset  size  field
    0       2     magic b'HV'
    2       1     version (1)
    3       1     kind (1 = audio)
    4       4     epoch      uint32 LE   -- bumped on every new turn AND every cancel
    8       4     seq        uint32 LE   -- frame index within the epoch
    12      4     sample_rate uint32 LE
    16      ...   float32 LE mono PCM
"""
import struct

MAGIC = b"HV"
VERSION = 1
KIND_AUDIO = 1
HEADER = struct.Struct("<2sBBIII")
HEADER_SIZE = HEADER.size   # 16


def pack_audio(epoch: int, seq: int, sample_rate: int, pcm_bytes: bytes) -> bytes:
    return HEADER.pack(MAGIC, VERSION, KIND_AUDIO, epoch & 0xFFFFFFFF,
                       seq & 0xFFFFFFFF, sample_rate) + pcm_bytes


def unpack_header(buf: bytes):
    magic, ver, kind, epoch, seq, sr = HEADER.unpack_from(buf, 0)
    if magic != MAGIC or ver != VERSION:
        raise ValueError("bad frame header")
    return kind, epoch, seq, sr


# ------------------------------------------------------------------ server -> client events
# {"type": "ready",      "config": {...}}
# {"type": "state",      "state": "listening|user_speaking|thinking|speaking"}
# {"type": "turn_start", "turn": int, "epoch": int, "barge_in": bool}
# {"type": "asr",        "turn": int, "text": str, "ms": float}
# {"type": "text",       "turn": int, "delta": str}          streaming brain output
# {"type": "sentence",   "turn": int, "text": str}           a sentence handed to TTS
# {"type": "cancel",     "epoch": int, "reason": "barge_in|turn_end|client"}
#        -> the client MUST immediately clear its playback buffer for that epoch
# {"type": "turn_end",   "turn": int, "epoch": int, "cancelled": bool}
# {"type": "metrics",    "turn": int, ...}
# {"type": "error",      "message": str}
#
# ------------------------------------------------------------------ client -> server control
# {"type": "hello",      "token": str, "sample_rate": int}
# {"type": "played",     "epoch": int, "seq": int}   playback ack, used for backpressure
# {"type": "stop"}
