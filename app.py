from flask import Flask, request, jsonify
import os
import requests

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

@app.route("/", methods=["GET"])
def home():
    return "ROBONEX AI Voice Server is running!"

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(silent=True) or {}
        prompt = data.get("text", "").strip()

        if not prompt:
            return jsonify({"error": "Please provide text"}), 400

        if not GEMINI_API_KEY:
            return jsonify({"error": "GEMINI_API_KEY is not configured"}), 500

        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/gemini-3.8-flash:generateContent"
        )

        response = requests.post(
            url,
            params={"key": GEMINI_API_KEY},
            json={
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            },
            timeout=40
        )

        result = response.json()

        if response.status_code != 200:
            return jsonify({"error": result}), response.status_code

        answer = result["candidates"][0]["content"]["parts"][0]["text"]
        return jsonify({"reply": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
