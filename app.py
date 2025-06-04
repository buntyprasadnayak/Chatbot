from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

app = Flask(__name__)
CORS(app)

model_name = "mistralai/Mistral-7B-Instruct-v0.1"

print("Loading Mistral model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": "cpu"},  # Runs on CPU
    torch_dtype=torch.float32  # Use float32 on CPU
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    prompt = f"[INST] {user_input} [/INST]"
    result = generator(prompt)[0]["generated_text"]
    return jsonify({"response": result.split('[/INST]')[-1].strip()})

@app.route("/", methods=["GET"])
def home():
    return "Mistral Chatbot Backend Running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
