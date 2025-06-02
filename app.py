from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

app = Flask(__name__)

# Loading the model (make sure to have sufficient GPU memory)
model_name = "tiiuae/falcon-7b-instruct"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,  # or float16 if needed
    trust_remote_code=True
)

# Use pipeline for easy generation
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
    prompt = f"User: {user_input}\nAssistant:"
    result = generator(prompt)[0]["generated_text"]
    # Remove the prompt from the result if needed
    response = result.split("Assistant:")[-1].strip()
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
