from flask import Flask, request, jsonify, Response
import os
import requests

# ==============================
# ROBONEX AI - Flask Server
# ==============================

app = Flask(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_URL = "https://api.groq.com/openai/v1"

CHAT_MODEL = "openai/gpt-oss-120b"
STT_MODEL = "whisper-large-v3-turbo"
TTS_MODEL = "canopylabs/orpheus-v1-english"


def groq_headers():
    return {
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }


def groq_error_response(feature, response):
    app.logger.error(
        "%s failed: HTTP %s - %s",
        feature,
        response.status_code,
        response.text[:2000]
    )

    return jsonify({
        "error": f"{feature} failed",
        "status": response.status_code,
        "details": response.text[:2000]
    }), 502


# ==============================
# HOME
# ==============================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "assistant": "ROBONEX AI",
        "status": "online",
        "message": "ROBONEX AI Voice Server is running!",
        "endpoints": [
            "/",
            "/health",
            "/ask",
            "/transcribe",
            "/speak"
        ]
    })


# ==============================
# HEALTH CHECK
# ==============================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "assistant": "ROBONEX AI",
        "status": "online",
        "api_key_configured": bool(GROQ_API_KEY)
    })


# ==============================
# AI CHAT
# POST /ask
# JSON: {"text": "Hello"}
# ==============================

@app.route("/ask", methods=["POST"])
def ask():
    if not GROQ_API_KEY:
        return jsonify({
            "error": "GROQ_API_KEY is missing in Render environment"
        }), 500

    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return jsonify({
            "error": "Please provide text"
        }), 400

    try:
        response = requests.post(
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
                            "You are ROBONEX, a friendly AI robot assistant. "
                            "Understand the user's question and answer "
                            "naturally, clearly, and concisely. "
                            "You can communicate in Hindi or English "
                            "depending on the user's language."
                        )
                    },
                    {
                        "role": "user",
                        "content": text.strip()
                    }
                ],
                "max_completion_tokens": 300
            },
            timeout=60
        )

        if not response.ok:
            return groq_error_response("Groq chat", response)

        result = response.json()
        answer = (
            result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        return jsonify({
            "assistant": "ROBONEX AI",
            "reply": answer.strip()
        })

    except requests.RequestException:
        app.logger.exception("Groq chat connection failed")
        return jsonify({
            "error": "Could not connect to Groq chat API"
        }), 502

    except (ValueError, KeyError, IndexError, TypeError):
        app.logger.exception("Invalid Groq chat response")
        return jsonify({
            "error": "Invalid response received from Groq"
        }), 502


# ==============================
# SPEECH TO TEXT
# POST /transcribe
# Multipart form field: audio
# ==============================

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if not GROQ_API_KEY:
        return jsonify({
            "error": "GROQ_API_KEY is missing in Render environment"
        }), 500

    audio = request.files.get("audio")

    if audio is None:
        return jsonify({
            "error": "Audio file is missing",
            "usage": "Send an audio file using multipart field 'audio'"
        }), 400

    if not audio.filename:
        return jsonify({
            "error": "Audio filename is missing"
        }), 400

    try:
        response = requests.post(
            f"{GROQ_URL}/audio/transcriptions",
            headers=groq_headers(),
            files={
                "file": (
                    audio.filename,
                    audio.stream,
                    audio.mimetype or "application/octet-stream"
                )
            },
            data={
                "model": STT_MODEL,
                "response_format": "json"
            },
            timeout=90
        )

        if not response.ok:
            return groq_error_response(
                "Groq transcription",
                response
            )

        result = response.json()

        return jsonify({
            "assistant": "ROBONEX AI",
            "text": result.get("text", "").strip()
        })

    except requests.RequestException:
        app.logger.exception("Transcription connection failed")
        return jsonify({
            "error": "Could not connect to Groq transcription API"
        }), 502

    except ValueError:
        app.logger.exception("Invalid transcription response")
        return jsonify({
            "error": "Invalid transcription response"
        }), 502


# ==============================
# TEXT TO SPEECH
# POST /speak
# JSON: {"text": "Hello ROBONEX"}
# Returns WAV audio
# ==============================

@app.route("/speak", methods=["POST"])
def speak():
    if not GROQ_API_KEY:
        return jsonify({
            "error": "GROQ_API_KEY is missing in Render environment"
        }), 500

    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return jsonify({
            "error": "Please provide text to speak"
        }), 400

    try:
        response = requests.post(
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
            timeout=90
        )

        if not response.ok:
            return groq_error_response(
                "Groq text-to-speech",
                response
            )

        return Response(
            response.content,
            status=200,
            content_type="audio/wav",
            headers={
                "Content-Disposition":
                    'inline; filename="robonex.wav"',
                "Cache-Control": "no-store"
            }
        )

    except requests.RequestException:
        app.logger.exception("Speech generation connection failed")
        return jsonify({
            "error": "Could not connect to Groq text-to-speech API"
        }), 502


# ==============================
# ERROR HANDLERS
# ==============================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found"
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "HTTP method not allowed for this endpoint"
    }), 405


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "Uploaded file is too large"
    }), 413


# ==============================
# LOCAL DEVELOPMENT
# Render uses Gunicorn
# ==============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
