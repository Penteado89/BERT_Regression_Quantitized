import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Importar o modelo e outras funções necessárias do arquivo training.py
from models.bert_models import BertForBalancedClassification  # Use o modelo para classificação balanceada
from src.utils import TextDataset
from Train.classification_task import define_class_labels

# Função para calcular métricas por classe
def calculate_metrics_by_class(real_values, predictions, num_classes):
    metrics = {}
    for i in range(num_classes):
        true_positive = sum((np.array(real_values) == i) & (np.array(predictions) == i))
        true_negative = sum((np.array(real_values) != i) & (np.array(predictions) != i))
        false_positive = sum((np.array(real_values) != i) & (np.array(predictions) == i))
        false_negative = sum((np.array(real_values) == i) & (np.array(predictions) != i))
        
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
        specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0

        metrics[i] = {
            'Precision': precision,
            'Recall': recall,
            'Specificity': specificity
        }
    return metrics

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
    precision = precision_score(real_values, predictions, average='weighted')
    recall = recall_score(real_values, predictions, average='weighted')
    f1 = f1_score(real_values, predictions, average='weighted')

    return avg_loss, accuracy, precision, recall, f1, real_values, predictions

# Função Principal para Avaliar Tarefa 3
def main():
    test_df = pd.read_csv('/content/drive/MyDrive/EP02_Regression_BERT/Data/test_set_10k.csv')
    test_df = define_class_labels(test_df)
    test_dataset = TextDataset(test_df['texto'], test_df['class'])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = '/content/drive/MyDrive/EP02_Regression_BERT/Data/model_best_val_loss_classification_task3.pt'
    model = BertForBalancedClassification()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    results = evaluate(model, test_loader, criterion, device)
    avg_loss, accuracy, precision, recall, f1, real_values, predictions = results  # Unpack all the values

    resultados_df = pd.DataFrame([[avg_loss, accuracy, precision, recall, f1]], columns=['Loss', 'Accuracy', 'Precision', 'Recall', 'F1'])

    # Print the classification report
    print("Classification Report:")
    print(classification_report(real_values, predictions))

    # Salva as métricas em um arquivo CSV
    resultados_df.to_csv('/content/drive/MyDrive/EP02_Regression_BERT/evaluation_metrics_classification_task3.csv', index=False)

if __name__ == "__main__":
    main()
