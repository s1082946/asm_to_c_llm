import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
import json

device = torch.device("cpu")

model_name = "/home/xchen/asm_to_c/models/codellama/CodeLlama-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, lora_config)

max_length = 2048
with open('processed_dataset_id', 'r') as f:
    data = json.load(f)[:1]  

class CodeDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = tokenizer(self.data[idx]['asm'], return_tensors="pt", padding="max_length", max_length=max_length)
        labels = tokenizer(self.data[idx]['Intrinsics'], return_tensors="pt", padding="max_length", max_length=max_length)
        inputs['labels'] = labels['input_ids']
        return {k: v.squeeze() for k, v in inputs.items()}

dataset = CodeDataset(data)
dataloader = DataLoader(dataset, batch_size=1)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=5e-5,
    logging_dir='./logs',
    logging_steps=10,
    gradient_checkpointing=False,
    no_cuda=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=lambda data: {
        'input_ids': torch.stack([f['input_ids'] for f in data]),
        'labels': torch.stack([f['labels'] for f in data]),
        'attention_mask': torch.stack([f['attention_mask'] for f in data]),
    }
)

trainer.train()

model.save_pretrained("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")
