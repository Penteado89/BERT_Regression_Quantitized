import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class TextDataset(Dataset):
    def __init__(self, texts, densities):
        self.tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
        self.texts = texts
        self.densities = densities

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        density = self.densities[idx]
        inputs = self.tokenizer(text, padding='max_length', truncation=True, 
                                max_length=512, return_tensors="pt")
        # Usar squeeze() para garantir a forma correta
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        return {'input_ids': input_ids, 'attention_mask': attention_mask}, density

class TextDataset2(Dataset):
    def __init__(self, texts, class_labels):
        self.tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
        self.texts = texts
        self.class_labels = class_labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        class_label = self.class_labels[idx]
        inputs = self.tokenizer(text, padding='max_length', truncation=True, 
                                max_length=512, return_tensors="pt")
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'class_label': torch.tensor(class_label, dtype=torch.long)
        }
