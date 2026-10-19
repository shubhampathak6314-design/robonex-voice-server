from flask import Flask, request, jsonify, Response
import os
import requests

app = Flask(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1"

CHAT_MODEL = "openai/gpt-oss-120b"
STT_MODEL = "whisper-large-v3-turbo"
TTS_MODEL = "canopylabs/orpheus-v1-english"


@app.route("/", methods=["GET"])
def home():
    return "ROBONEX AI Voice Server is running!"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "assistant": "ROBONEX AI"
    })


def groq_headers():
    return {
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }


@app.route("/ask", methods=["POST"])
def ask():
    if not GROQ_API_KEY:
        return jsonify({"error": "Missing GROQ_API_KEY"}), 500

    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Enter text"}), 400

    try:
        r = requests.post(
            f"{GROQ_URL}/chat/completions",
            headers={
                **groq_headers(),
                "Content-Type": "application/json"
            },
            json={
                "model": CHAT_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are ROBONEX, a friendly robot assistant. "
                            "Reply naturally and concisely."
                        )
                    },
                    {"role": "user", "content": text.strip()}
                ],
                "max_completion_tokens": 300
            },
            timeout=45
        )

        if not r.ok:
            return jsonify({
                "error": "Groq chat failed",
                "details": r.text[:1000]
            }), 502

        answer = r.json()["choices"][0]["message"]["content"].strip()

        return jsonify({
            "assistant": "ROBONEX AI",
            "reply": answer
        })

    except requests.RequestException:
        app.logger.exception("Groq request failed")
        return jsonify({"error": "AI connection failed"}), 502


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if not GROQ_API_KEY:
        return jsonify({"error": "Missing GROQ_API_KEY"}), 500

    audio = request.files.get("audio")

    if not audio:
        return jsonify({"error": "Upload an audio file"}), 400

    try:
        r = requests.post(
            f"{GROQ_URL}/audio/transcriptions",
            headers=groq_headers(),
            files={
                "file": (
                    audio.filename or "recording.wav",
                    audio.stream,
                    audio.mimetype or "audio/wav"
                )
            },
            data={
                "model": STT_MODEL,
                "response_format": "json"
            },
            timeout=60
        )

        if not r.ok:
            return jsonify({
                "error": "Transcription failed",
                "details": r.text[:1000]
            }), 502

        return jsonify({"text": r.json().get("text", "")})

    except requests.RequestException:
        app.logger.exception("Transcription request failed")
        return jsonify({"error": "Transcription connection failed"}), 502


@app.route("/speak", methods=["POST"])
def speak():
    if not GROQ_API_KEY:
        return jsonify({"error": "Missing GROQ_API_KEY"}), 500

    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Enter text"}), 400

    try:
        r = requests.post(
            f"{GROQ_URL}/audio/speech",
            headers={
                **groq_headers(),
                "Content-Type": "application/json"
            },
            json={
                "model": TTS_MODEL,
                "voice": "austin",
                "input": text.strip()[:200],
                "response_format": "wav"
            },
            timeout=60
        )

        if not r.ok:
            return jsonify({
                "error": "Speech generation failed",
                "details": r.text[:1000]
            }), 502

        return Response(
            r.content,
            mimetype="audio/wav",
            headers={"Cache-Control": "no-store"}
        )

    except requests.RequestException:
        app.logger.exception("Speech request failed")
        return jsonify({"error": "Speech connection failed"}), 502


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
