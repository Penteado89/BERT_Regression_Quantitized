import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Importar o modelo e outras funções necessárias do arquivo training.py
from models.bert_models import BertForBalancedClassification  # Use o modelo para classificação balanceada
from src.utils import TextDataset
from Train.classification_task import define_class_labels

# Função para calcular métricas por classe
def calculate_specificity(y_true, y_pred, class_id):
    cm = confusion_matrix(y_true, y_pred)
    true_negatives = cm.sum() - cm[:, class_id].sum() - cm[class_id, :].sum() + cm[class_id, class_id]
    false_positives = cm[:, class_id].sum() - cm[class_id, class_id]
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    return specificity

# Função de avaliação
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss, predictions, real_values = 0, [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            data_dict, class_label = batch
            input_ids = data_dict['input_ids'].to(device)
            attention_mask = data_dict['attention_mask'].to(device)
            class_label = class_label.to(device)

            # Obter apenas os logits da saída do modelo
            outputs = model(input_ids, attention_mask)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            loss = criterion(logits, class_label)
            total_loss += loss.item()

            predictions.extend(logits.argmax(dim=1).cpu().numpy())
            real_values.extend(class_label.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(real_values, predictions)

    # Uso de zero_division e labels corretos
    precision = precision_score(real_values, predictions, average='weighted', labels=[0, 1, 2], zero_division=0)
    recall = recall_score(real_values, predictions, average='weighted', labels=[0, 1, 2], zero_division=0)
    f1 = f1_score(real_values, predictions, average='weighted', labels=[0, 1, 2])

    cm = confusion_matrix(real_values, predictions)
    class_accuracies = [(cm[i, i] / cm[i, :].sum()) if cm[i, :].sum() > 0 else 0 for i in range(3)]
    class_recalls = [recall_score(real_values, predictions, labels=[i], average='weighted', zero_division=0) for i in range(3)]
    class_specificities = [calculate_specificity(real_values, predictions, i) for i in range(3)]

    return avg_loss, accuracy, precision, recall, f1, class_accuracies, class_recalls, class_specificities, real_values, predictions

# Função Principal para Avaliar Tarefa 3
def main():
    test_df = pd.read_csv('/content/drive/MyDrive/EP02_Regression_BERT/Data/test_set_10k.csv')
    test_df = define_class_labels(test_df)
    test_dataset = TextDataset(test_df['texto'], test_df['class'])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = '/content/drive/MyDrive/EP02_Regression_BERT/Data/model_best_val_loss_classification_task3.pt'
    model = BertForBalancedClassification()
    criterion = torch.nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    results = evaluate(model, test_loader, criterion, device)
    avg_loss, accuracy, precision, recall, f1, class_accuracies, class_recalls, class_specificities, _, _ = results

    resultados_df = pd.DataFrame({
        'Accuracy': [accuracy],
        'Class 0 Accuracy': [class_accuracies[0]],
        'Class 1 Accuracy': [class_accuracies[1]],
        'Class 2 Accuracy': [class_accuracies[2]],
        'Class 0 Recall': [class_recalls[0]],
        'Class 1 Recall': [class_recalls[1]],
        'Class 2 Recall': [class_recalls[2]],
        'Class 0 Specificity': [class_specificities[0]],
        'Class 1 Specificity': [class_specificities[1]],
        'Class 2 Specificity': [class_specificities[2]]
    })
    display(resultados_df)
    # Salva as métricas em um arquivo CSV
    resultados_df.to_csv('/content/drive/MyDrive/EP02_Regression_BERT/evaluation_metrics_classification_task3.csv', index=False)

if __name__ == "__main__":
    main()
