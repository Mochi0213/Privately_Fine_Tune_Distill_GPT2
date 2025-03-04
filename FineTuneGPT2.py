import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset


dataset = load_dataset("text", data_files="processed_data.txt")
train_dataset = dataset["train"]

tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length = 256)

train_dataset = train_dataset.map(tokenize_function, batched=True)

model = GPT2LMHeadModel.from_pretrained("distilgpt2")


training_args = TrainingArguments(
    output_dir="./model",
    evaluation_strategy="no",
    save_strategy="epoch",
    num_train_epochs=100,
    per_device_train_batch_size=10,
    warmup_steps=1000,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=10,
    learning_rate=3e-5,
    disable_tqdm=False
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer = tokenizer,
    mlm = False
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)


trainer.train()


model.save_pretrained("fine_tuned_gpt2")
tokenizer.save_pretrained("fine_tuned_gpt2")

print("Train Completed, Fine Tuned Model parameters have been stored in fine_tuned_gpt2")

