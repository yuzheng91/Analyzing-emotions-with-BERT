import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn, optim
from tqdm import tqdm
from dataset import AnalysisDataset
from model import Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # Use bert-base-uncased as our pretrained model

def read_and_tokenize(file_path):
    with open(file_path, encoding='utf-8') as f:
        text = f.read()
    encoded_dict = tokenizer.encode_plus(
        text,                      
        add_special_tokens=True,   
        max_length=512,            
        pad_to_max_length=True,    
        return_attention_mask=True,
        return_tensors='pt',       
        truncation=True            
    )
    return encoded_dict

def process_reviews(directory_path, tokenizer):
    input_ids = []
    attention_masks = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            encoded_dict = read_and_tokenize(file_path)
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat([x for x in input_ids], dim=0)
    attention_masks = torch.cat([x for x in attention_masks], dim=0)
    
    return {'input_ids': input_ids, 'attention_mask': attention_masks}

neg_train_path = 'aclImdb/train/neg'
neg_test_path = 'aclImdb/test/neg'
neg_train_tokenized = process_reviews(neg_train_path, tokenizer)
neg_test_tokenized = process_reviews(neg_test_path, tokenizer)

pos_train_path = 'aclImdb/train/pos'
pos_test_path = 'aclImdb/test/pos'
pos_train_tokenized = process_reviews(pos_train_path, tokenizer)
pos_test_tokenized = process_reviews(pos_test_path, tokenizer)

train_labels = [0] * len(neg_train_tokenized['input_ids']) + [1] * len(pos_train_tokenized['input_ids'])
test_labels = [0] * len(neg_test_tokenized['input_ids']) + [1] * len(pos_test_tokenized['input_ids'])

train_tokenized_texts = {
    'input_ids': torch.cat([neg_train_tokenized['input_ids'], pos_train_tokenized['input_ids']], dim=0),
    'attention_mask': torch.cat([neg_train_tokenized['attention_mask'], pos_train_tokenized['attention_mask']], dim=0)
}
test_tokenized_texts = {
    'input_ids': torch.cat([neg_test_tokenized['input_ids'], pos_test_tokenized['input_ids']], dim=0),
    'attention_mask': torch.cat([neg_test_tokenized['attention_mask'], pos_test_tokenized['attention_mask']], dim=0)
}

train_dataset = AnalysisDataset(train_tokenized_texts, train_labels)
test_dataset = AnalysisDataset(test_tokenized_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

model = Classifier()
model.to(device)

model_save_path = 'model_save'

epochs = 4
total_steps = len(train_loader) * epochs
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps) #optimize lr

loss_fn = nn.CrossEntropyLoss().to(device)

model.train()

best_val_accuracy = 0.0

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    print('-' * 10)
    
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}  # Use GPU to compute

        optimizer.zero_grad()

        # forward pass
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        loss = loss_fn(outputs, batch['labels'])

        # backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    # compute average loss
    avg_train_loss = total_loss / len(train_loader)
    print(f"Average Training Loss: {avg_train_loss:.4f}")

    model.eval()

    all_preds = []
    all_labels = []
    
    correct_predictions = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = {k: v.to(device) for k, v in batch.items()}  # Use GPU to compute
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            preds = torch.argmax(outputs, dim=1)
            labels = batch['labels']
            correct_predictions += (preds == labels).long().sum()
            all_preds.extend(preds)
            all_labels.extend(labels)
            
    all_preds = torch.stack(all_preds)
    all_labels = torch.stack(all_labels)
    all_preds = all_preds.cpu().numpy()
    all_labels = all_labels.cpu().numpy()
    accuracy = correct_predictions.double() / len(test_dataset)
    if accuracy>best_val_accuracy:
        best_val_accuracy = accuracy
        save_path = os.path.join(model_save_path, f'model1')
        torch.save(model.state_dict(), save_path)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision: .4f}")
    print(f"Recall: {recall: .4f}")
    print(f"F1-Score: {f1: .4f}")

model.eval()