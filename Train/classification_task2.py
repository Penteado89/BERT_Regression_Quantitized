import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm  
import matplotlib.pyplot as plt  
from transformers import BertModel, BertTokenizer
from src.utils import TextDataset2
from models.bert_models import BertForQuantizedClassification

def define_class_labels(df, threshold_low, threshold_high):
    """
    Define class labels based on vowel density thresholds.
    """
    conditions = [
        df['vowel_density'] < threshold_low,
        df['vowel_density'].between(threshold_low, threshold_high),
        df['vowel_density'] > threshold_high
    ]
    df['class'] = np.select(conditions, [0, 1, 2])
    return df

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for d in tqdm(train_loader, desc="Training"):
        input_ids, attention_mask, class_label = (d['input_ids'].to(device), 
                                                  d['attention_mask'].to(device), 
                                                  d['class_label'].to(device))

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, class_label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss, predictions, real_values = 0, [], []
    with torch.no_grad():
        for d in tqdm(val_loader, desc="Evaluating"):
            input_ids, attention_mask, class_label = (d['input_ids'].to(device), 
                                                      d['attention_mask'].to(device), 
                                                      d['class_label'].to(device))
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, class_label)
            total_loss += loss.item()
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            real_values.extend(class_label.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(real_values, predictions)
    precision = precision_score(real_values, predictions, average='weighted')
    recall = recall_score(real_values, predictions, average='weighted')
    f1 = f1_score(real_values, predictions, average='weighted')

    return avg_loss, accuracy, precision, recall, f1

def main():
    train_df = pd.read_csv('/content/drive/MyDrive/EP02_Regression_BERT/Data/train_set_10k.csv')
    val_df = pd.read_csv('/content/drive/MyDrive/EP02_Regression_BERT/Data/val_set_10k.csv')

    train_df = define_class_labels(train_df, 1/3, 2/3)
    val_df = define_class_labels(val_df, 1/3, 2/3)

    train_dataset = TextDataset2(train_df['texto'], train_df['class'])
    val_dataset = TextDataset2(val_df['texto'], val_df['class'])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForQuantizedClassification().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(10):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, accuracy, precision, recall, f1 = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = '/content/drive/MyDrive/EP02_Regression_BERT/Data/model_best_val_loss_classification.pt'
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch} with validation loss: {best_val_loss}")

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
