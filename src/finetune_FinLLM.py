import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from sklearn.model_selection import train_test_split
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Go4miii/DISC-FinLLM", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Go4miii/DISC-FinLLM",
    torch_dtype=torch.float16,  # Mixed precision for speed and memory efficiency
    trust_remote_code=True,
    device_map="auto"          # Automatically map model to GPU
)
# model = model.eval()

# Configure LoRA for efficient fine-tuning
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["W_pack"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

file_path = "./dataset.json"

# Load and split dataset
with open(file_path, "r", encoding="utf8") as f:
    datas = json.load(f)

filtered_data = [item for item in datas if item.get('domain') == '经济金融']

train_data, test_data = train_test_split(filtered_data, test_size=0.3, random_state=42)

# Define prompt-building function
def build_prompt(q, q_type):
    if q_type == "单选题":
        return f'这是一道{q_type}，请给出你的答案（只需给出单个选项），并进行解析，给出你的理由。\n' + \
                f'题目：{q}\n' + \
                '答案：'
    elif q_type == "多选题":
        return f'这是一道{q_type}，请给出你的答案（只需给出选项），并进行解析，给出你的理由。\n' + \
                f'题目：{q}\n' + \
                '答案：'
    elif q_type == "判断题":
        return f'这是一道{q_type}，请给出你的答案（只需给出“正确”或“错误”），并进行解析，给出你的理由。\n' + \
                f'题目：{q}\n' + \
                '答案：'
    raise ValueError(f"Unsupported question type: {q_type}")

# Format examples
def format_example(item):
    prompt = build_prompt(item["question"], item["question_type"])
    answer = item["answer"]
    return {"text": prompt, "label": answer}

# Prepare datasets
train_dataset = Dataset.from_list([format_example(item) for item in train_data])
test_dataset = Dataset.from_list([format_example(item) for item in test_data])

# Tokenization function
def tokenize_function(examples):
    prompts = examples["text"]
    answers = examples["label"]
    full_texts = [p + a for p, a in zip(prompts, answers)]
    tokenized = tokenizer(full_texts, padding="max_length", truncation=True, max_length=512)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    prompt_lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    labels = []
    for ids, p_len in zip(input_ids, prompt_lengths):
        label = [-100] * p_len + ids[p_len:]
        label = label[:512] if len(label) >= 512 else label + [-100] * (512 - len(label))
        labels.append(label)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# Tokenize datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text", "label"])
tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=["text", "label"])

training_args = TrainingArguments(
    output_dir="./finetuned_model",
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,
    num_train_epochs=5,  # Increased epochs
    learning_rate=1e-5,  # Slightly lower learning rate
    fp16=True,
    save_strategy="steps",
    save_steps=20,
    logging_steps=1,
    report_to="none",
    evaluation_strategy="no",  # Changed to periodically evaluate
    dataloader_num_workers=4,  # Increase workers if possible
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",  # Changed to cosine for better convergence
    optim="adamw_torch",  # Explicitly use AdamW for stability
)


# Initialize trainer with explicit optimizer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    optimizers=(torch.optim.AdamW(model.parameters(), lr=5e-5), None)
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")