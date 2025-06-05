# prever_novo_xgboost.py

import os
import pandas as pd
import joblib

# Importar o dicionário de mapeamento reverso
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from mapeamento_classes import reverse_mapping_dict

# Caminho correto para o modelo
modelo_path = os.path.join(base_dir, "output", "modelos", "modelo_xgboost_inicial_mapeado.pkl")
modelo = joblib.load(modelo_path)

# 2. Criar um novo exemplo de aluno (com as mesmas colunas usadas no treino!)
novo_aluno = pd.DataFrame([{
    'Módulo atual': 1,
    'Faltas Consecutivas': 0,
    'Histórico de reprovações': 0,
    'Histórico de Recuperação': 0,
    'Historico de Reprovado por Falta (disciplinas)': 0,
    'Idade': 22,
    'Sexo (código)': 0, 
    'Pend. Acad.': 0,
    'Possui Pendência Financeira': 0,
    'Bolsista': 0,
    'Antecipou Parcela': 1
}])

# 3. Fazer a previsão
predicao = modelo.predict(novo_aluno)
codigo = int(predicao[0])
nome_situacao = reverse_mapping_dict.get(codigo, "Desconhecido")

# 4. Mostrar o resultado
print(f'Previsão da Situação (código): {codigo}')
print(f'Previsão da Situação (nome): {nome_situacao}')
