import os
import yaml
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, # CHANGED: Specific for T5
    Seq2SeqTrainer,           # CHANGED: Specific for T5
    DataCollatorForSeq2Seq    # CHANGED: Handles padding correctly
)
from peft import LoraConfig, get_peft_model, TaskType

# Load config
with open("training/lora_config.yaml", "r") as f:
    config = yaml.safe_load(f)

BASE_MODEL = config["base_model"]
OUTPUT_DIR = config["output_dir"]

# Load dataset
dataset = load_dataset(
    "json",
    data_files="data/raw/ecommerce_qa.jsonl",
    split="train"
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

# --- FIX 1: MATCH DATASET COLUMNS ---
# --- FIX 1: MATCH DATASET COLUMNS ---
def format_example(example):
    # DYNAMIC INPUT FORMATTING
    # If context exists and is not empty, add it. Otherwise just use instruction.
    context_text = example.get('context', "")
    
    if context_text:
        input_text = (
            f"instruction: {example['instruction']}\n"
            f"context: {context_text}\n"
            f"response:"
        )
    else:
        input_text = (
            f"instruction: {example['instruction']}\n"
            f"response:"
        )

    target_text = example["response"]

    model_inputs = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=config["training"]["max_seq_length"],
    )
    # ... rest of the function remains the same ... 
    labels = tokenizer(
        target_text,
        truncation=True,
        padding="max_length",
        max_length=config["training"]["max_seq_length"],
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
# ------------------------------------

dataset = dataset.map(format_example, remove_columns=dataset.column_names)

# LoRA config
lora_config = LoraConfig(
    r=config["lora"]["r"],
    lora_alpha=config["lora"]["lora_alpha"],
    lora_dropout=config["lora"]["lora_dropout"],
    target_modules=config["lora"]["target_modules"],
    bias=config["lora"]["bias"],
    task_type=TaskType.SEQ_2_SEQ_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- FIX 2: USE SEQ2SEQ TRAINER ---
# This ensures padding tokens are ignored in loss calculation
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100
)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=config["training"]["num_train_epochs"],
    per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
    gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
    learning_rate=float(config["training"]["learning_rate"]), # Cast to float just in case
    warmup_steps=config["training"]["warmup_steps"],
    logging_steps=config["training"]["logging_steps"],
    fp16=config["training"]["fp16"],
    save_strategy=config["training"]["save_strategy"],
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator, # Add the collator
)
# ----------------------------------

trainer.train()

# Save adapters only
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
