
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath):
    # Carregar o conjunto de dados
    df = pd.read_csv(filepath)

    # Manter apenas a coluna 'review_text' e renomeá-la para 'texto'
    df = df[['review_text']].rename(columns={'review_text': 'texto'})
    
    df = df.head(10000)

    return df

# Função para calcular a densidade de vogais
def calculate_vowel_density(text):
    vowels = "aeiouAEIOUáéíóúÁÉÍÓÚàèìòùÀÈÌÒÙãõÃÕ"
    letters = [char for char in text if char.isalpha()]
    vowel_count = sum(char in vowels for char in letters)
    return vowel_count / len(letters) if letters else 0

def split_data(df):
    # Dividir os dados em treinamento, validação e teste (60%, 20%, 20%)
    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    return train_df, val_df, test_df

def save_dataframes(train_df, val_df, test_df):
    train_df.to_csv('/content/drive/MyDrive/EP02_Regression_BERT/Data/train_set_10k.csv', index=False)
    val_df.to_csv('/content/drive/MyDrive/EP02_Regression_BERT/Data/val_set_10k.csv', index=False)
    test_df.to_csv('/content/drive/MyDrive/EP02_Regression_BERT/Data/test_set_10k.csv', index=False)

def main():
    # Substitua 'your_dataset.csv' pelo caminho do seu arquivo de dados
    filepath = '/content/drive/MyDrive/EP02_Regression_BERT/Data/B2W-Reviews01.csv'
    df = load_and_preprocess_data(filepath)
    df = df.dropna(subset=['texto'])
    df['vowel_density'] = df['texto'].apply(calculate_vowel_density)
    train_df, val_df, test_df = split_data(df)
    save_dataframes(train_df, val_df, test_df)

if __name__ == "__main__":
    main()
