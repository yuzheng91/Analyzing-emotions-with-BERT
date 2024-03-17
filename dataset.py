import torch
from torch.utils.data import DataLoader, Dataset

class AnalysisDataset(Dataset):
    def __init__(self, encoded_texts, labels):
        self.encoded_texts = encoded_texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encoded_texts.items()}
        item['labels'] = self.labels[idx]
        return item