import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig
import sys
import os

# 1. Setup Paths
ADAPTER_PATH = "adapters/flan_t5_ecommerce_lora"
BASE_MODEL_ID = "google/flan-t5-small"

# Safety Check
if not os.path.exists(ADAPTER_PATH):
    print(f"❌ Error: Adapter path not found at {ADAPTER_PATH}")
    print("   Did you unzip the 'adapters.zip' from Colab?")
    sys.exit(1)

print("⏳ Loading Base Model (this may take a moment)...")
# 2. Load Base Model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_ID)

print(f"🔗 Loading LoRA Adapters from {ADAPTER_PATH}...")
# 3. Load Adapters
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval() # Switch to evaluation mode

print("\n✅ Model Loaded! Type 'quit' to exit.\n")
print("-" * 50)

# 4. Chat Loop
while True:
    try:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        
        # Prepare Prompt (Matching the training format)
        prompt = f"instruction: {user_input}\nresponse:"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate
        # Generate with clearer constraints
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True, 
                temperature=0.3,      # LOWER this (was 0.7). Makes it less random/creative.
                top_p=0.9,
                repetition_penalty=1.2, # ADD this. Stops "stylized stylized stylized"
                num_beams=3           # ADD this. Looks for the "best" path, not just a random one.
            )
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Bot: {response}")
        print("-" * 50)
        
    except KeyboardInterrupt:
        break

print("\n👋 Chat closed.")