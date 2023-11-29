import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F

class BertForVowelDensityRegression(nn.Module):
    """
    Modelo BERT para a tarefa de regressão da densidade de vogais.
    """
    def __init__(self, model_name='neuralmind/bert-base-portuguese-cased'):
        super(BertForVowelDensityRegression, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        # Uma camada linear para regressão
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        # Use squeeze to remove the extra dimension
        return self.regressor(pooled_output).squeeze()

class BertForQuantizedClassification(nn.Module):
    """
    Modelo BERT para a tarefa de classificação quantizada.
    Inclui dropout para regularização e uma camada linear para classificação.
    """
    def __init__(self, model_name='neuralmind/bert-base-portuguese-cased', num_classes=3):
        super(BertForQuantizedClassification, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)

        # Adicionando dropout para regularização
        self.dropout = nn.Dropout(0.1)

        # Camada de classificação
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # Passando os inputs através do modelo BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Utilizando o pooled_output para tarefas de classificação
        pooled_output = outputs[1]

        # Aplicando dropout ao pooled_output
        pooled_output = self.dropout(pooled_output)

        # Retornando a saída da camada de classificação
        return self.classifier(pooled_output)

class BertForBalancedClassification(nn.Module):
    """
    Modelo BERT para classificação com classes balanceadas.
    """
    def __init__(self, model_name='neuralmind/bert-base-portuguese-cased', num_classes=3):
        super(BertForBalancedClassification, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        
        # Adicione uma camada softmax para calcular as probabilidades de classe
        logits = self.classifier(pooled_output)
        probabilities = F.softmax(logits, dim=-1)
        
        return logits, probabilities
