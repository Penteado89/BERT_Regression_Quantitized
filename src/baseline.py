# Importações
import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from scipy.stats import pearsonr
import numpy as np

# Importar o modelo e outras funções necessárias do arquivo training.py
from models.bert_models import BertForVowelDensityRegression
from scr.preprocess_and_split import calculate_vowel_density
from scr.utils import TextDataset


def load_and_preprocess_data(filepath):
    # Carregar o conjunto de dados
    df = pd.read_csv(filepath)

    # Manter apenas a coluna 'review_text' e renomeá-la para 'texto'
    df = df[['review_text']].rename(columns={'review_text': 'texto'})
    
    df = df.head(10000)

    return df

def calcular_metricas(y_real, y_pred):
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae = mean_absolute_error(y_real, y_pred)
    mape = mean_absolute_percentage_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    
    valid_indices = ~np.isnan(y_real) & ~np.isnan(y_pred)
    pearson_corr, _ = pearsonr(y_real[valid_indices], y_pred[valid_indices])

    return rmse, mae, mape, r2, pearson_corr

def densidade_vogais_primeira_palavra(sentenca):
    primeira_palavra = sentenca.split()[0]
    return calculate_vowel_density(primeira_palavra)

def densidade_vogais_ultima_palavra(sentenca):
    ultima_palavra = sentenca.split()[-1]
    return calculate_vowel_density(ultima_palavra)

def avaliar_baseline(df):
    # Cálculo da densidade média do corpus como baseline
    densidade_media_corpus = df['vowel_density'].mean()
    df['densidade_corpus'] = np.full_like(df['vowel_density'], densidade_media_corpus)

    # Aplicação das funções de densidade de vogais
    df['densidade_primeira_palavra'] = df['texto'].apply(densidade_vogais_primeira_palavra)
    df['densidade_ultima_palavra'] = df['texto'].apply(densidade_vogais_ultima_palavra)

    resultados = []

    for baseline in ['densidade_corpus', 'densidade_primeira_palavra', 'densidade_ultima_palavra']:
        y_pred = df[baseline]
        y_real = df['vowel_density']
        rmse, mae, mape, r2, pearson_corr = calcular_metricas(y_real, y_pred)
        
        resultados.append({
            'Baseline': baseline,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'Pearson Correlation': pearson_corr
        })

    resultados_df = pd.DataFrame(resultados)
    return resultados_df


# Função Principal para Teste
def main():
    filepath = '/content/drive/MyDrive/EP02_Regression_BERT/Data/B2W-Reviews01.csv'
    df = load_and_preprocess_data(filepath)
    df = df.dropna(subset=['texto'])
    df['vowel_density'] = df['texto'].apply(calculate_vowel_density)
    # Carregar dados de teste

    # Avaliar os baselines
    resultados_baseline = avaliar_baseline(df)
    display(resultados_baseline)
    resultados_baseline.to_csv('/content/drive/MyDrive/EP02_Regression_BERT/baseline.csv', index=False)

if __name__ == "__main__":
    main()
