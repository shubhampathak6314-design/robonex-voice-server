from flask import Flask, request, jsonify
import os
import requests

app = Flask(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

MODEL = "llama-3.3-70b-versatile"


@app.route("/", methods=["GET"])
def home():
    return "ROBONEX AI Voice Server is running!"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "assistant": "ROBONEX AI"
    })


@app.route("/ask", methods=["POST"])
def ask():
    try:
        if not GROQ_API_KEY:
            return jsonify({
                "error": "GROQ_API_KEY is not configured in Render"
            }), 500

        data = request.get_json(silent=True) or {}

        prompt = data.get("text", "")

        if not isinstance(prompt, str) or not prompt.strip():
            return jsonify({
                "error": "Please provide text in the 'text' field"
            }), 400

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are ROBONEX, a friendly AI voice assistant. "
                        "You understand Hindi, Hinglish, and English. "
                        "Reply in the same language the user uses. "
                        "Keep answers natural, clear, and concise, "
                        "because your answers may be spoken aloud. "
                        "You are helpful, polite, and friendly."
                    )
                },
                {
                    "role": "user",
                    "content": prompt.strip()
                }
            ],
            "temperature": 0.7,
            "max_completion_tokens": 500
        }

        response = requests.post(
            GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=45
        )

        try:
            result = response.json()
        except ValueError:
            return jsonify({
                "error": "Invalid response received from AI provider"
            }), 502

        if response.status_code != 200:
            return jsonify({
                "error": "Groq API request failed",
                "details": result
            }), 502

        answer = (
            result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        if not answer:
            return jsonify({
                "error": "AI returned an empty response"
            }), 502

        return jsonify({
            "reply": answer.strip(),
            "assistant": "ROBONEX AI"
        })

    except requests.Timeout:
        return jsonify({
            "error": "AI response timed out. Please try again."
        }), 504

    except requests.RequestException:
        app.logger.exception("Groq API connection failed")
        return jsonify({
            "error": "Could not connect to the AI service"
        }), 502

    except Exception:
        app.logger.exception("Unexpected server error")
        return jsonify({
            "error": "Internal server error"
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
