#!/usr/bin/env sh
# Entrypoint switch for the shared GPU image: `asr` or `tts`.
set -eu
case "${1:-asr}" in
  asr) exec /venv/bin/python -m hervoice.svc.asr_service ;;
  tts) exec /venv/bin/python -m hervoice.svc.tts_service ;;
  shell) exec /bin/sh ;;
  *) echo "usage: entrypoint {asr|tts|shell}" >&2; exit 2 ;;
esac
