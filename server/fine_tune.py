import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, 
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
import os

# Load model & tokenizer
MODEL_NAME = "facebook/opt-1.3b"  # Alternative: "mistralai/Mistral-7B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# Ensure correct dataset path
dataset_path = os.path.abspath("./training_data.jsonl")
dataset = load_dataset("json", data_files={"train": dataset_path})

# Split dataset into training and evaluation
dataset = dataset["train"].train_test_split(test_size=0.2)

# Tokenization function
def tokenize_function(examples):
    inputs = [f"Rephrase the following sentence, '{text}', to be understandable by an autistic person playing a game" for text in examples["input"]]
    outputs = examples["output"]
    tokenized_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=50, return_tensors="pt")
    tokenized_outputs = tokenizer(outputs, padding="max_length", truncation=True, max_length=50, return_tensors="pt")
    tokenized_inputs["labels"] = tokenized_outputs["input_ids"]  # Set labels for causal LM
    return tokenized_inputs

# Apply tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["input", "output"])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./model_finetuned",
    eval_strategy="epoch",  # Perform evaluation at every epoch
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
)

# Trainer with eval_dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

# Train the model with error handling
try:
    trainer.train()
    trainer.save_model("./trained_model")
    tokenizer.save_pretrained("./trained_model")
    print("✅ Fine-tuning complete! Model saved in './trained_model'")
except Exception as e:
    print(f"❌ Error during fine-tuning: {e}")
