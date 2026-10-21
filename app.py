import asyncio, io, os, re, subprocess, threading, wave
import requests
from flask import Flask, Response, jsonify, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
GROQ_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_BASE = "https://api.groq.com/openai/v1"
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "whisper-large-v3")
WHISPER_LANG = os.environ.get("WHISPER_LANG", "").strip()
RETRY_LANG = os.environ.get("RETRY_LANG", "hi").strip() or "hi"
ALLOWED_LANGS = {"en", "english", "hi", "hindi"}
last_audio = b""
audio_lock = threading.Lock()
DEFAULT_LLMS = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile",
 "meta-llama/llama-4-scout-17b-16e-instruct", "openai/gpt-oss-20b", "qwen/qwen3-32b"]
LLM_MODELS = []
for m in [os.environ.get("LLM_MODEL", "").strip()] + DEFAULT_LLMS:
    if m and m not in LLM_MODELS: LLM_MODELS.append(m)
working_model = None
TTS_ORDER = [x.strip().lower() for x in os.environ.get("TTS_ORDER", "edge,groq").split(",") if x.strip()]
EDGE_VOICE = os.environ.get("EDGE_VOICE", "en-US-AriaNeural")
HINDI_VOICE = os.environ.get("HINDI_VOICE", "hi-IN-SwaraNeural")
GROQ_TTS_MODEL = os.environ.get("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english")
GROQ_TTS_VOICE = os.environ.get("GROQ_TTS_VOICE", "autumn")
TTS_RATE = 24000
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
HALLUCINATIONS = {"thank you","thanks","thank you very much","thanks for watching","thank you for watching",
 "you","bye","bye bye","okay","ok","hmm","the","a","i","so","uh","um","oh","yeah"}
SYSTEM_PROMPT = ("You are ROBONEX AI, a friendly voice assistant inside a small robot. Reply in the same "
 "language the user spoke. If Hindi or Hinglish, reply in Hindi written in Devanagari. If English, reply "
 "in English. Answer in 1 to 3 short sentences, plain text only. No markdown, lists, emojis, or asterisks.")
history, MAX_HISTORY = [], 6
history_lock = threading.Lock()

def clean_text(s):
    s = re.sub(r"<think>.*?</think>", " ", str(s or ""), flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"[*#`_~]", "", s)).strip()

def pcm_to_wav(pcm, rate):
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm)
    return out.getvalue()

async def edge_mp3(text):
    import edge_tts
    voice = HINDI_VOICE if DEVANAGARI.search(text) else EDGE_VOICE
    out = bytearray()
    async for item in edge_tts.Communicate(text, voice).stream():
        if item.get("type") == "audio": out.extend(item["data"])
    return bytes(out)

def tts_edge(text):
    import imageio_ffmpeg
    mp3 = asyncio.run(edge_mp3(text))
    if not mp3: raise RuntimeError("Edge TTS returned no audio")
    p = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
      "-i", "pipe:0", "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(TTS_RATE),
      "-ac", "1", "pipe:1"], input=mp3, capture_output=True, timeout=60)
    if p.returncode or not p.stdout:
        raise RuntimeError("ffmpeg failed: " + p.stderr.decode("utf-8","ignore")[:200])
    return pcm_to_wav(p.stdout, TTS_RATE)

def tts_groq(text):
    if not GROQ_KEY: raise RuntimeError("GROQ_API_KEY is not configured")
    r = requests.post(GROQ_BASE+"/audio/speech", headers={"Authorization":"Bearer "+GROQ_KEY},
      json={"model":GROQ_TTS_MODEL,"input":text,"voice":GROQ_TTS_VOICE,"response_format":"wav"}, timeout=60)
    if r.status_code != 200: raise RuntimeError("Groq TTS %d: %s"%(r.status_code,r.text[:200]))
    if not r.content.startswith(b"RIFF"): raise RuntimeError("Groq TTS did not return WAV")
    return r.content

def run_stt(data, lang=None):
    if not GROQ_KEY: return None, "GROQ_API_KEY is not configured in Render environment"
    fields = {"model":WHISPER_MODEL,"response_format":"verbose_json","temperature":"0"}
    if lang: fields["language"] = lang
    try:
        r = requests.post(GROQ_BASE+"/audio/transcriptions", headers={"Authorization":"Bearer "+GROQ_KEY},
          files={"file":("voice.wav",data,"audio/wav")}, data=fields, timeout=60)
    except requests.RequestException as e: return None, str(e)
    if r.status_code != 200: return None, "%d %s"%(r.status_code,r.text[:300])
    try: return r.json(), ""
    except ValueError: return None, "Whisper returned invalid JSON"

def looks_like_noise(j):
    segs = j.get("segments") or []
    if not segs: return False
    if all((s.get("no_speech_prob") or 0) > .7 for s in segs): return True
    lp = [s.get("avg_logprob") for s in segs if s.get("avg_logprob") is not None]
    return bool(lp) and sum(lp)/len(lp) < -1.5

def call_llm(msgs):
    global working_model
    if not GROQ_KEY: return None, "GROQ_API_KEY is not configured in Render environment"
    order = ([working_model] if working_model else []) + [m for m in LLM_MODELS if m != working_model]
    last = "no model tried"
    for m in order:
        payload = {"model":m,"messages":msgs,"temperature":.6,"max_tokens":300}
        if m.startswith("openai/gpt-oss"): payload["reasoning_effort"] = "low"
        try:
            r = requests.post(GROQ_BASE+"/chat/completions", headers={"Authorization":"Bearer "+GROQ_KEY},
                              json=payload, timeout=30)
        except requests.RequestException as e:
            last = f"{m}: {e}"; continue
        if r.status_code == 200:
            try: content = r.json()["choices"][0]["message"].get("content") or ""
            except (ValueError,KeyError,IndexError,TypeError): content = ""
            reply = clean_text(content)
            if reply: working_model = m; return reply, ""
            last = f"{m}: empty reply"; continue
        last = f"{m} {r.status_code}: {r.text[:220]}"
        app.logger.warning("LLM request failed: %s", last)
        if r.status_code == 401: break
        if working_model == m: working_model = None
    return None, last

@app.route("/", methods=["GET","HEAD"])
def root(): return "ROBONEX server OK", 200

@app.route("/health")
def health():
    return jsonify({"ok":True,"groq_key_configured":bool(GROQ_KEY),"stt_model":WHISPER_MODEL,"llm_models":LLM_MODELS})

@app.route("/transcribe", methods=["POST"])
def transcribe():
    global last_audio
    f = request.files.get("audio")
    if f is None: return jsonify({"error":"missing multipart field 'audio'"}), 400
    data = f.read()
    if len(data) < 1000: return jsonify({"text":""})
    if not data.startswith(b"RIFF") or data[8:12] != b"WAVE":
        return jsonify({"error":"audio must be a valid WAV file"}), 400
    with audio_lock: last_audio = data
    j, err = run_stt(data, WHISPER_LANG or None)
    if j is None: return jsonify({"error":"stt failed","details":err}), 502
    lang = str(j.get("language") or "").lower()
    if not WHISPER_LANG and lang and lang not in ALLOWED_LANGS:
        j2, err2 = run_stt(data, RETRY_LANG)
        if j2 is not None: j = j2
        else: app.logger.warning("STT retry failed: %s", err2)
    text = (j.get("text") or "").strip()
    if looks_like_noise(j): text = ""
    if re.sub(r"[^a-z ]","",text.lower()).strip() in HALLUCINATIONS: text = ""
    app.logger.info("STT lang=%s text=%r",j.get("language"),text)
    return jsonify({"text":text})

@app.route("/last.wav")
def last_wav():
    with audio_lock: data = last_audio
    if not data: return "No audio recorded yet", 404
    return Response(data,mimetype="audio/wav")

@app.route("/ask", methods=["POST"])
def ask():
    body = request.get_json(silent=True) or {}
    q = (body.get("text") or "").strip()
    if not q: return jsonify({"error":"empty text"}), 400
    with history_lock: msgs = [{"role":"system","content":SYSTEM_PROMPT}] + history[-MAX_HISTORY:] + [{"role":"user","content":q}]
    reply, err = call_llm(msgs)
    if not reply: return jsonify({"error":"llm failed","details":err}), 502
    with history_lock:
        history.extend([{"role":"user","content":q},{"role":"assistant","content":reply}])
        del history[:-MAX_HISTORY]
    return jsonify({"reply":reply})

@app.route("/speak", methods=["POST"])
def speak():
    body = request.get_json(silent=True) or {}
    text = clean_text(body.get("text",""))[:400]
    if not text: return jsonify({"error":"empty text"}), 400
    errors = []
    for name in TTS_ORDER:
        fn = {"edge":tts_edge,"groq":tts_groq}.get(name)
        if not fn or (name=="groq" and DEVANAGARI.search(text)): continue
        try: return Response(fn(text),mimetype="audio/wav",headers={"Cache-Control":"no-store"})
        except Exception as e:
            errors.append(f"{name}: {e}"); app.logger.warning("TTS %s failed: %s",name,e)
    return jsonify({"error":"all TTS providers failed","details":errors}), 502

@app.errorhandler(413)
def too_large(_e): return jsonify({"error":"audio upload too large"}), 413

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT","5000")))
