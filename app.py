# ============================================================
#  ROBONEX AI - voice server (Render)
#  /            -> health check (server jagane ke liye)
#  /transcribe  -> WAV audio (form field "audio") -> {"text": "..."}
#  /ask         -> {"text": "..."} -> {"reply": "..."}
#  /speak       -> {"text": "..."} -> WAV audio (16-bit mono)
#  TTS: Edge TTS (free, no limit) + Groq Orpheus (fallback)
# ============================================================
import asyncio
import io
import os
import re
import subprocess
import wave

import requests
from flask import Flask, Response, jsonify, request

app = Flask(__name__)

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE = "https://api.groq.com/openai/v1"

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "whisper-large-v3-turbo")
WHISPER_LANG = os.environ.get("WHISPER_LANG", "en")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")

# TTS: "edge" pehle, fail ho to "groq". Order badalna ho to env me TTS_ORDER=groq,edge
TTS_ORDER = [p.strip() for p in os.environ.get("TTS_ORDER", "edge,groq").split(",") if p.strip()]
EDGE_VOICE = os.environ.get("EDGE_VOICE", "en-US-AriaNeural")
GROQ_TTS_MODEL = os.environ.get("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english")
GROQ_TTS_VOICE = os.environ.get("GROQ_TTS_VOICE", "autumn")
TTS_RATE = 24000   # ESP32 SPEAKER_RATE se match

SYSTEM_PROMPT = (
    "You are ROBONEX AI, a friendly voice assistant inside a small robot. "
    "Answer in 1 to 3 short sentences, plain text only. "
    "No markdown, no lists, no emojis, no asterisks."
)

history = []          # last few turns (ek hi device hai)
MAX_HISTORY = 6

# Whisper silence par aksar yahi bakwas likh deta hai - inhe ignore karo
HALLUCINATIONS = {
    "thank you", "thanks", "thank you very much", "thanks for watching",
    "thank you for watching", "you", "bye", "bye bye", "okay", "ok", "hmm",
}


def _ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def pcm_to_wav(pcm, rate):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def clean_text(s):
    s = re.sub(r"<think>.*?</think>", " ", s, flags=re.S)
    s = re.sub(r"[*#`_~]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ---------------- TTS providers ----------------
async def _edge_mp3(text):
    import edge_tts
    comm = edge_tts.Communicate(text, EDGE_VOICE)
    buf = bytearray()
    async for ch in comm.stream():
        if ch["type"] == "audio":
            buf += ch["data"]
    return bytes(buf)


def tts_edge(text):
    mp3 = asyncio.run(_edge_mp3(text))
    if not mp3:
        raise RuntimeError("edge-tts returned no audio")
    p = subprocess.run(
        [_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
         "-i", "pipe:0", "-f", "s16le", "-acodec", "pcm_s16le",
         "-ar", str(TTS_RATE), "-ac", "1", "pipe:1"],
        input=mp3, capture_output=True, timeout=60,
    )
    if p.returncode != 0 or not p.stdout:
        raise RuntimeError("ffmpeg failed: " + p.stderr.decode("utf-8", "ignore")[:200])
    return pcm_to_wav(p.stdout, TTS_RATE)


def tts_groq(text):
    r = requests.post(
        GROQ_BASE + "/audio/speech",
        headers={"Authorization": "Bearer " + GROQ_KEY},
        json={"model": GROQ_TTS_MODEL, "input": text,
              "voice": GROQ_TTS_VOICE, "response_format": "wav"},
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError("groq tts %d: %s" % (r.status_code, r.text[:200]))
    if r.content[:4] != b"RIFF":
        raise RuntimeError("groq tts: not a WAV")
    return r.content


PROVIDERS = {"edge": tts_edge, "groq": tts_groq}


# ---------------- Routes ----------------
@app.route("/", methods=["GET", "HEAD"])
def root():
    return "ROBONEX server OK"


@app.route("/transcribe", methods=["POST"])
def transcribe():
    f = request.files.get("audio")
    if f is None:
        return jsonify({"error": "no audio field"}), 400
    data = f.read()
    if len(data) < 1000:
        return jsonify({"text": ""})

    r = requests.post(
        GROQ_BASE + "/audio/transcriptions",
        headers={"Authorization": "Bearer " + GROQ_KEY},
        files={"file": ("voice.wav", data, "audio/wav")},
        data={"model": WHISPER_MODEL, "response_format": "json",
              "language": WHISPER_LANG, "temperature": "0"},
        timeout=60,
    )
    if r.status_code != 200:
        return jsonify({"error": "stt failed", "details": r.text[:300]}), 502

    text = (r.json().get("text") or "").strip()
    norm = re.sub(r"[^a-z ]", "", text.lower()).strip()
    if norm in HALLUCINATIONS:
        text = ""
    return jsonify({"text": text})


@app.route("/ask", methods=["POST"])
def ask():
    body = request.get_json(force=True, silent=True) or {}
    q = (body.get("text") or "").strip()
    if not q:
        return jsonify({"error": "empty text"}), 400

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-MAX_HISTORY:]
    msgs.append({"role": "user", "content": q})

    r = requests.post(
        GROQ_BASE + "/chat/completions",
        headers={"Authorization": "Bearer " + GROQ_KEY},
        json={"model": LLM_MODEL, "messages": msgs,
              "temperature": 0.6, "max_tokens": 200},
        timeout=60,
    )
    if r.status_code != 200:
        return jsonify({"error": "llm failed", "details": r.text[:300]}), 502

    reply = clean_text(r.json()["choices"][0]["message"]["content"] or "")
    if not reply:
        return jsonify({"error": "empty reply"}), 502

    history.append({"role": "user", "content": q})
    history.append({"role": "assistant", "content": reply})
    del history[:-MAX_HISTORY]
    return jsonify({"reply": reply})


@app.route("/speak", methods=["POST"])
def speak():
    body = request.get_json(force=True, silent=True) or {}
    text = clean_text(body.get("text") or "")[:400]
    if not text:
        return jsonify({"error": "empty text"}), 400

    errors = []
    for name in TTS_ORDER:
        fn = PROVIDERS.get(name)
        if not fn:
            continue
        try:
            wav = fn(text)
            return Response(wav, mimetype="audio/wav")
        except Exception as e:
            errors.append("%s: %s" % (name, e))
            app.logger.warning("TTS %s failed: %s", name, e)

    return jsonify({"error": "all TTS failed", "details": errors}), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
