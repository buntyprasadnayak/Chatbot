from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

# === CONFIG ===
# API_KEY = "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" 
 # 🔁 Replace this with your real OpenRouter key

API_KEY = "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"
MODEL_NAME = "mistralai/mistral-small-3.2-24b-instruct:free"  # You can change to openai/gpt-3.5-turbo

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# === ROUTES ===
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": user_input}]
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload)
        result = response.json()
        message = result["choices"][0]["message"]["content"]
        return jsonify({"response": message})
    except Exception as e:
        return jsonify({"response": f"⚠️ Error: {str(e)}"})

@app.route("/", methods=["GET"])
def home():
    return "✅ OpenRouter Chatbot Backend Running!"

# === MAIN ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
