import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset
import os
import json


# Constants
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
DATA_PATH = "/app/actionaparsnip_mails.json"
MAX_LENGTH = 512
BATCH_SIZE = 1  # Reduced for memory constraints
EPOCHS = 3
LEARNING_RATE = 2e-4
HF_TOKEN = "##################################"

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model():
    # Use 4-bit quantization to save GPU memory
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        use_auth_token=HF_TOKEN
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def preprocess_data(json_path, tokenizer):
    with open(json_path, "r") as f:
        data = json.load(f)[:10]  # Trim for dev/debug

    dataset = Dataset.from_list(data)

    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["input"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )
        labels = tokenizer(
            examples["response"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(tokenize_function, batched=True)

def train_model(model, tokenizer, dataset):
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="no",
        save_steps=100,
        logging_steps=10,
        label_names=["labels"],
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        save_total_limit=1,
        fp16=True,
        gradient_checkpointing=True,  # Save memory
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    torch.cuda.empty_cache()  # Free up memory before training
    trainer.train()

# Execution
tokenizer = load_tokenizer()
model = load_model()
dataset = preprocess_data(DATA_PATH, tokenizer)
os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
train_model(model, tokenizer, dataset)
