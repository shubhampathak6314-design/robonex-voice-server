from flask import Flask, request, jsonify, Response
import asyncio
import os
import subprocess

import edge_tts
import imageio_ffmpeg
import requests

app = Flask(__name__)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_URL = "https://api.groq.com/openai/v1"
CHAT_MODEL = "openai/gpt-oss-120b"
STT_MODEL = "whisper-large-v3-turbo"
TTS_VOICE = "hi-IN-SwaraNeural"


def groq_headers():
    return {"Authorization": f"Bearer {GROQ_API_KEY}"}


def groq_error_response(feature, response):
    app.logger.error("%s failed: HTTP %s - %s", feature, response.status_code, response.text[:2000])
    return jsonify({"error": f"{feature} failed", "status": response.status_code,
                    "details": response.text[:2000]}), 502


@app.route("/", methods=["GET"])
def home():
    return jsonify({"assistant": "ROBONEX AI", "status": "online",
                    "message": "ROBONEX AI Voice Server is running!", "tts": "Edge TTS",
                    "endpoints": ["/", "/health", "/ask", "/transcribe", "/speak"]})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"assistant": "ROBONEX AI", "status": "online",
                    "api_key_configured": bool(GROQ_API_KEY), "tts": "edge-tts"})


@app.route("/ask", methods=["POST"])
def ask():
    if not GROQ_API_KEY:
        return jsonify({"error": "GROQ_API_KEY is missing in Render environment"}), 500
    data = request.get_json(silent=True) or {}
    user_text = data.get("text", "")
    if not isinstance(user_text, str) or not user_text.strip():
        return jsonify({"error": "Please provide text"}), 400
    try:
        response = requests.post(
            f"{GROQ_URL}/chat/completions",
            headers={**groq_headers(), "Content-Type": "application/json"},
            json={"model": CHAT_MODEL, "messages": [
                {"role": "system", "content": "You are ROBONEX, a friendly AI robot assistant. Answer naturally, clearly, and concisely. Reply in Hindi or English according to the user's language."},
                {"role": "user", "content": user_text.strip()}],
                "max_completion_tokens": 300},
            timeout=60)
        if not response.ok:
            return groq_error_response("Groq chat", response)
        result = response.json()
        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return jsonify({"assistant": "ROBONEX AI", "reply": answer.strip()})
    except requests.RequestException:
        app.logger.exception("Groq chat connection failed")
        return jsonify({"error": "Could not connect to Groq chat API"}), 502
    except (ValueError, KeyError, IndexError, TypeError):
        app.logger.exception("Invalid Groq chat response")
        return jsonify({"error": "Invalid response received from Groq"}), 502


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if not GROQ_API_KEY:
        return jsonify({"error": "GROQ_API_KEY is missing in Render environment"}), 500
    audio = request.files.get("audio")
    if audio is None or not audio.filename:
        return jsonify({"error": "Audio file is missing",
                        "usage": "Send an audio file using multipart field 'audio'"}), 400
    try:
        response = requests.post(
            f"{GROQ_URL}/audio/transcriptions",
            headers=groq_headers(),
            files={"file": (audio.filename, audio.stream, audio.mimetype or "application/octet-stream")},
            data={"model": STT_MODEL, "response_format": "json"},
            timeout=90)
        if not response.ok:
            return groq_error_response("Groq transcription", response)
        result = response.json()
        return jsonify({"assistant": "ROBONEX AI", "text": result.get("text", "").strip()})
    except requests.RequestException:
        app.logger.exception("Transcription connection failed")
        return jsonify({"error": "Could not connect to Groq transcription API"}), 502
    except (ValueError, TypeError):
        app.logger.exception("Invalid transcription response")
        return jsonify({"error": "Invalid transcription response"}), 502


async def edge_tts_mp3(text):
    communicator = edge_tts.Communicate(text, TTS_VOICE)
    chunks = []
    async for chunk in communicator.stream():
        if chunk.get("type") == "audio":
            chunks.append(chunk["data"])
    if not chunks:
        raise RuntimeError("Edge TTS returned no audio")
    return b"".join(chunks)


def mp3_to_wav(mp3_data):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
               "-f", "wav", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "24000", "pipe:1"]
    result = subprocess.run(command, input=mp3_data, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=30, check=False)
    if result.returncode != 0 or not result.stdout.startswith(b"RIFF"):
        app.logger.error("MP3 to WAV conversion failed: %s",
                         result.stderr.decode("utf-8", "replace")[:1000])
        raise RuntimeError("Audio conversion failed")
    return result.stdout


@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json(silent=True) or {}
    text_to_speak = data.get("text", "")
    if not isinstance(text_to_speak, str) or not text_to_speak.strip():
        return jsonify({"error": "Please provide text to speak"}), 400
    try:
        mp3_data = asyncio.run(edge_tts_mp3(text_to_speak.strip()[:200]))
        wav_data = mp3_to_wav(mp3_data)
        return Response(wav_data, status=200, content_type="audio/wav",
                        headers={"Content-Disposition": 'inline; filename="robonex.wav"',
                                 "Cache-Control": "no-store"})
    except Exception:
        app.logger.exception("Edge TTS generation/conversion failed")
        return jsonify({"error": "Could not generate speech with Edge TTS",
                        "hint": "Check Render logs and confirm edge-tts/imageio-ffmpeg installed successfully"}), 502


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "HTTP method not allowed for this endpoint"}), 405


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({"error": "Uploaded file is too large"}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
