import math
import torch
from torch.optim import Adam
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from utils.dp_optimizer import make_optimizer_class

dataset = load_dataset("text", data_files="processed_data.txt")
train_dataset = dataset["train"]

tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)


train_dataset = train_dataset.map(tokenize_function, batched=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)

training_args = TrainingArguments(
    output_dir="./model",
    evaluation_strategy="no",
    save_strategy="epoch",
    num_train_epochs=200,
    per_device_train_batch_size=10,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    learning_rate=1e-4,
    disable_tqdm=False,
)


def compute_epsilon(steps, batch_size, dataset_size, sigma, delta=1e-2):
    sampling_probability = batch_size / dataset_size
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]

    def compute_rdp(sampling_probability, sigma, steps, orders):
        return [steps * sampling_probability ** 2 / (2 * sigma ** 2) for order in orders]

    rdp = compute_rdp(sampling_probability, sigma, steps, orders)
    epsilon = min([rdp[i] / (orders[i] - 1) for i in range(len(orders))])

    return epsilon


optimizer = make_optimizer_class(Adam)
optimizer = optimizer(
    l2_norm_clip = 0.1,
    noise_multiplier=1.0,
    minibatch_size=training_args.per_device_train_batch_size,
    microbatch_size=1,
    params = model.parameters(),
    lr = training_args.learning_rate
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    optimizers=(optimizer, None)
)


trainer.train()

steps = math.ceil(len(train_dataset) / training_args.per_device_train_batch_size) * training_args.num_train_epochs
epsilon = compute_epsilon(steps, training_args.per_device_train_batch_size, len(train_dataset), optimizer.noise_multiplier)
print(f"Final (ε, δ)-DP = ({epsilon:.2f}, 1e-2)")

model.save_pretrained("fine_tuned_gpt2_dp")
tokenizer.save_pretrained("fine_tuned_gpt2_dp")

print("Train Completed, Fine Tuned Model parameters have been stored in fine_tuned_gpt2_dp")
