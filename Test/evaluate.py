# Importações
import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from scipy.stats import pearsonr
import numpy as np

# Importar o modelo e outras funções necessárias do arquivo training.py
from models.bert_models import BertForVowelDensityRegression
from src.preprocess_and_split import calculate_vowel_density
from src.utils import TextDataset

def calcular_metricas(y_real, y_pred):
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae = mean_absolute_error(y_real, y_pred)
    mape = mean_absolute_percentage_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    
    valid_indices = ~np.isnan(y_real) & ~np.isnan(y_pred)
    pearson_corr, _ = pearsonr(y_real[valid_indices], y_pred[valid_indices])

    return rmse, mae, mape, r2, pearson_corr

# Avaliação do modelo no conjunto de teste
def evaluate(model, test_loader, device):
    model.eval()
    y_pred = []
    y_real = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move inputs and labels to the correct device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs).squeeze()  # Ensure outputs match expected format
            logits = outputs.cpu().numpy()  # Move outputs back to CPU for numpy conversion
            y_pred.extend(logits)
            y_real.extend(labels.cpu().numpy())  # Move labels back to CPU for numpy conversion
    return y_pred, y_real  # Adicione esta linha

# Função Principal para Teste
def main():
    # Carregar dados de teste
    test_df = pd.read_csv('/content/drive/MyDrive/EP02_Regression_BERT/Data/test_set_10k.csv')
    
    test_dataset = TextDataset(test_df['texto'], test_df['vowel_density'])

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carregar modelo treinado
    model = BertForVowelDensityRegression(model_name='neuralmind/bert-base-portuguese-cased')
    model_path = '/content/drive/MyDrive/EP02_Regression_BERT/Data/model_best_val_loss.pt'

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        # Load the model on the CPU
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    modelo = 'Regression_BERT'
    # Avaliação do Modelo
    y_pred, y_real = evaluate(model, test_loader, device)  # Capture os valores retornados
    rmse, mae, mape, r2, pearson_corr = calcular_metricas(np.array(y_real), np.array(y_pred))
    
    resultados_df = pd.DataFrame({
            'Baseline': modelo,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'Pearson Correlation': pearson_corr
        }, index=[0])


    # Exibir o DataFrame
    display(resultados_df)
    resultados_df.to_csv('/content/drive/MyDrive/EP02_Regression_BERT/evaluation_metrics_regression.csv', index=False)

if __name__ == "__main__":
    main()
