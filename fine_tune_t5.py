#!/usr/bin/python3

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

# Load the WikiHow dataset
train_articles_file = 'articles.txt'
train_summaries_file = 'summaries.txt'

# Load the T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_accumulation_steps=10,
    learning_rate=5e-5,
    evaluation_strategy='steps',
    save_total_limit=2,
    eval_steps=200,
    save_steps=200,
    warmup_steps=500,
    logging_dir='./logs',
    logging_steps=200,
    load_best_model_at_end=True
)

# Define the data collator


def data_collator(features):
    input_ids = [f['input_ids'] for f in features]
    attention_mask = [f['attention_mask'] for f in features]
    labels = [f['input_ids'] for f in features]
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


class WikiHowDataset(torch.utils.data.Dataset):
    def init(self, tokenizer, articles_file, summaries_file):
        self.articles = []
        self.summaries = []
        self.tokenizer = tokenizer
        with open(articles_file, 'r', encoding='utf-8') as fa, open(summaries_file, 'r', encoding='utf-8') as fs:
            for article, summary in zip(fa, fs):
                self.articles.append(article.strip())
                self.summaries.append(summary.strip())

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        article = self.articles[idx]
        summary = self.summaries[idx]
        inputs = self.tokenizer.encode_plus(
            article, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        targets = self.tokenizer.encode_plus(
            summary, max_length=150, padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze(),
        }


train_dataset = WikiHowDataset(
    tokenizer, train_articles_file, train_summaries_file)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=2, shuffle=True, collate_fn=data_collator)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
output_dir = './models/'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)