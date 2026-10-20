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

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "whisper-large-v3")
# Khali = auto-detect (Hindi + English dono). Sirf Hindi: "hi", sirf English: "en"
WHISPER_LANG = os.environ.get("WHISPER_LANG", "").strip()
# LLM: jo model aapke Groq account me chale wahi use hoga (upar se neeche try karta hai).
# Pasandida model force karna ho to Render env me LLM_MODEL=model-name daalo.
_DEFAULT_LLMS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
]
LLM_MODELS = []
for _m in [os.environ.get("LLM_MODEL", "").strip()] + _DEFAULT_LLMS:
    if _m and _m not in LLM_MODELS:
        LLM_MODELS.append(_m)
working_model = None   # jo model chal gaya, usse yaad rakho

# TTS: "edge" pehle, fail ho to "groq". Order badalna ho to env me TTS_ORDER=groq,edge
TTS_ORDER = [p.strip() for p in os.environ.get("TTS_ORDER", "edge,groq").split(",") if p.strip()]
EDGE_VOICE = os.environ.get("EDGE_VOICE", "en-US-AriaNeural")
HINDI_VOICE = os.environ.get("HINDI_VOICE", "hi-IN-SwaraNeural")
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
GROQ_TTS_MODEL = os.environ.get("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english")
GROQ_TTS_VOICE = os.environ.get("GROQ_TTS_VOICE", "autumn")
TTS_RATE = 24000   # ESP32 SPEAKER_RATE se match

SYSTEM_PROMPT = (
    "You are ROBONEX AI, a friendly voice assistant inside a small robot. "
    "Reply in the same language the user spoke. If the user speaks Hindi or Hinglish, "
    "reply in Hindi written in Devanagari script. If the user speaks English, reply in English. "
    "Answer in 1 to 3 short sentences, plain text only. "
    "No markdown, no lists, no emojis, no asterisks."
)

history = []          # last few turns (ek hi device hai)
MAX_HISTORY = 6

# Whisper silence par aksar yahi bakwas likh deta hai - inhe ignore karo
HALLUCINATIONS = {
    "thank you", "thanks", "thank you very much", "thanks for watching",
    "thank you for watching", "you", "bye", "bye bye", "okay", "ok", "hmm",
    "the", "a", "i", "so", "uh", "um", "oh", "yeah",
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
    voice = HINDI_VOICE if DEVANAGARI.search(text) else EDGE_VOICE
    comm = edge_tts.Communicate(text, voice)
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


def stt_fields():
    d = {"model": WHISPER_MODEL, "response_format": "json", "temperature": "0"}
    if WHISPER_LANG:
        d["language"] = WHISPER_LANG
    return d


# ---------------- LLM with model fallback ----------------
def call_llm(msgs):
    """Models ko order me try karta hai. Return: (reply, error_text)"""
    global working_model
    order = ([working_model] if working_model else []) + \
            [m for m in LLM_MODELS if m != working_model]
    last = "no model tried"
    for m in order:
        payload = {"model": m, "messages": msgs, "temperature": 0.6, "max_tokens": 300}
        if m.startswith("openai/gpt-oss"):
            payload["reasoning_effort"] = "low"
        try:
            r = requests.post(
                GROQ_BASE + "/chat/completions",
                headers={"Authorization": "Bearer " + GROQ_KEY},
                json=payload, timeout=25,
            )
        except requests.RequestException as e:
            last = "%s: %s" % (m, e)
            continue

        if r.status_code == 200:
            try:
                content = r.json()["choices"][0]["message"].get("content") or ""
            except Exception:
                content = ""
            reply = clean_text(content)
            if reply:
                working_model = m
                return reply, ""
            last = "%s: empty reply" % m
            continue

        last = "%s %d: %s" % (m, r.status_code, r.text[:200])
        app.logger.warning("LLM %s", last)
        if r.status_code == 401:       # galat API key - baaki models bhi fail honge
            break
        if working_model == m:
            working_model = None
    return None, last


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
        data=stt_fields(),
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

    reply, err = call_llm(msgs)
    if not reply:
        return jsonify({"error": "llm failed", "details": err}), 502

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
        if name == "groq" and DEVANAGARI.search(text):
            continue            # Orpheus sirf English bolta hai
        try:
            wav = fn(text)
            return Response(wav, mimetype="audio/wav")
        except Exception as e:
            errors.append("%s: %s" % (name, e))
            app.logger.warning("TTS %s failed: %s", name, e)

    return jsonify({"error": "all TTS failed", "details": errors}), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
