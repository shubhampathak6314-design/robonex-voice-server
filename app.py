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
                "Content-Disposition": 'inline; filename="robonex.wav"',
                "Cache-Control": "no-store"
            }
        )

    except requests.RequestException:
        app.logger.exception("Groq TTS connection failed")
        return jsonify({
            "error": "Could not connect to Groq text-to-speech API"
        }), 502
