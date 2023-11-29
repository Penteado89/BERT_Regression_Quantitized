
# Importações
from transformers import BertModel, BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from models.bert_models import BertForVowelDensityRegression
from src.utils import TextDataset
from tqdm import tqdm


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        # Ensure that inputs and labels are on the correct device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(**inputs).squeeze()  # Squeeze the output to match label shape
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
          
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Ensure that inputs and labels are on the correct device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device).float()

            outputs = model(**inputs).squeeze()  # Squeeze the output to match label shape
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    return avg_val_loss

    print(f"Validation loss for epoch {epoch}: {avg_val_loss}")

def main():
    # Carregar e preparar dados de treino e validação
    train_df = pd.read_csv('/content/drive/MyDrive/EP02_Regression_BERT/Data/train_set_10k.csv')
    val_df = pd.read_csv('/content/drive/MyDrive/EP02_Regression_BERT/Data/val_set_10k.csv')

    train_dataset = TextDataset(train_df['texto'], train_df['vowel_density'])
    val_dataset = TextDataset(val_df['texto'], val_df['vowel_density'])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Inicializar modelo e critérios
    model = BertForVowelDensityRegression(model_name='neuralmind/bert-base-portuguese-cased')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    num_epochs = 10

    patience = 5  # Número de épocas para esperar após a última melhoria
    num_epochs_without_improvement = 0
    # Inicializar a melhor perda de validação
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # Treinar e validar o modelo
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train(model, tqdm(train_loader, desc="Training"), optimizer, criterion, device)
        val_loss = evaluate(model, tqdm(val_loader, desc="Evaluating"), criterion, device)  # Add device argument
        train_losses.append(train_loss)
        print(f'Train Loss: {train_loss}')
        val_losses.append(val_loss)
        print(f'Val Loss: {val_loss}')
        print(f'Epoch {epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}')

        scheduler.step(val_loss)

    # Verifica se houve melhoria
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            num_epochs_without_improvement = 0
            model_save_path = '/content/drive/MyDrive/EP02_Regression_BERT/Data/model_best_val_loss.pt'
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch} with validation loss: {best_val_loss}")
        else:
            num_epochs_without_improvement += 1

        # Early stopping
        if num_epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break


# Plotar as perdas
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.scatter(best_epoch, best_val_loss, color='red', marker='*', s=100, label=f'Best Epoch: {best_epoch}')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
