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
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "canopylabs/orpheus-v1-english",
                "voice": "austin",
                "input": text.strip()[:200],
                "response_format": "wav"
            },
            timeout=60
        )

        if not r.ok:
            app.logger.error(
                "Groq TTS failed: status=%s body=%s",
                r.status_code,
                r.text[:2000]
            )
            return jsonify({
                "error": "Groq TTS failed",
                "status": r.status_code,
                "details": r.text[:2000]
            }), 502

        return Response(
            r.content,
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
            "error": "Could not connect to Groq TTS"
        }), 502
