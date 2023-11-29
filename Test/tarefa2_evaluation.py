# Importações
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from models.bert_models import BertForQuantizedClassification  # Classe para classificação
from src.utils import TextDataset
from Train.classification_task2 import define_class_labels

# Avaliação do modelo no conjunto de teste
def evaluate(model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss, predictions, real_values = 0, [], []
    
    if torch.cuda.is_available():
        model.to(device)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Extract input_ids, attention_mask, and class_label from the batch
            data, class_label = batch
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            class_label = class_label.to(device)

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

    return avg_loss, accuracy, precision, recall, f1, real_values, predictions
    
# Função Principal para 

def main():
    test_df = pd.read_csv('/content/drive/MyDrive/EP02_Regression_BERT/Data/test_set_10k.csv')
    test_df = define_class_labels(test_df, 1/3, 2/3)
    test_dataset = TextDataset(test_df['texto'], test_df['class'])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = '/content/drive/MyDrive/EP02_Regression_BERT/Data/model_best_val_loss_classification.pt'
    model = BertForQuantizedClassification()
    criterion = torch.nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

    results = evaluate(model, test_loader, criterion, device)
    avg_loss, accuracy, precision, recall, f1, real_values, predictions = results  # Unpack all the values

    resultados_df = pd.DataFrame([[avg_loss, accuracy, precision, recall, f1]], columns=['Loss', 'Accuracy', 'Precision', 'Recall', 'F1'])

    # Print the classification report
    print("Classification Report:")
    print(classification_report(real_values, predictions))

    resultados_df.to_csv('/content/drive/MyDrive/EP02_Regression_BERT/evaluation_metrics_classification.csv', index=False)

if __name__ == "__main__":
    main()
