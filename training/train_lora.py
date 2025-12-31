import os
import yaml
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
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

# Format function
def format_example(example):
    input_text = (
        f"instruction: {example['instruction']}\n"
        f"question: {example['question']}\n"
        f"answer:"
    )
    target_text = example["answer"]

    model_inputs = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=config["training"]["max_seq_length"],
    )

    labels = tokenizer(
        target_text,
        truncation=True,
        padding="max_length",
        max_length=config["training"]["max_seq_length"],
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

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

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=config["training"]["num_train_epochs"],
    per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
    gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
    learning_rate=config["training"]["learning_rate"],
    warmup_steps=config["training"]["warmup_steps"],
    logging_steps=config["training"]["logging_steps"],
    fp16=config["training"]["fp16"],
    save_strategy=config["training"]["save_strategy"],
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

# Save adapters only
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

