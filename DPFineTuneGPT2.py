import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from private_transformers import PrivacyEngine
from torch import nn
import torch.nn.functional as F


dataset = load_dataset("text", data_files="processed_data.txt")
train_dataset = dataset["train"]

tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)


train_dataset = train_dataset.map(tokenize_function,
                                  batched=True,
                                  remove_columns=["text"],)

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
    tokenizer=tokenizer,
    mlm=False
)

train_dataloader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size,
                              collate_fn=data_collator)

optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)

privacy_engine = PrivacyEngine(
    model,
    batch_size=training_args.per_device_train_batch_size,
    sample_size=len(train_dataset),
    epochs=training_args.num_train_epochs,
    max_grad_norm=0.1,
    target_epsilon=1,
)
privacy_engine.attach(optimizer)
criterion = nn.CrossEntropyLoss()

for epoch in range(training_args.num_train_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{training_args.num_train_epochs}"):
        optimizer.zero_grad()
        outputs = model(**batch)

        labels = batch['input_ids'][:, 1:, ]
        logits = outputs.logits[:, :-1, :].permute(0, 2, 1)

        loss = F.cross_entropy(logits, labels, reduction="none").mean(dim=1)
        optimizer.step(loss=loss)
        total_loss += loss.mean().item()

    print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_dataloader)}")

model.save_pretrained("fine_tuned_gpt2")
tokenizer.save_pretrained("fine_tuned_gpt2")

print("Train Completed, Fine Tuned Model parameters have been stored in fine_tuned_gpt2")
