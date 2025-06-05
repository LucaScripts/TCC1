# prever_em_lote_xgboost.py

import pandas as pd
import joblib
import sys
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from mapeamento_classes import reverse_mapping_dict

# 1. Carregar o modelo treinado
modelo_path = os.path.join(base_dir, "output", "modelos", "modelo_xgboost_otimizado_mapeado.pkl")
modelo = joblib.load(modelo_path)  # Carrega o modelo XGBoost treinado

# 2. Carregar os dados dos alunos para prever
alunos_para_prever = pd.read_csv(os.path.join(base_dir, 'alunos_para_prever.csv'), encoding='latin1', sep=';')  # Arquivo de alunos

# 3. Garantir que as colunas correspondam às usadas no treinamento
# (As colunas do CSV já devem estar corretas)

# 4. Fazer as previsões
predicoes = modelo.predict(alunos_para_prever)

# 5. Adicionar as previsões ao DataFrame
alunos_para_prever['Previsão (Situação)'] = predicoes

# 6. Adicionar também a sigla correspondente
alunos_para_prever['Previsão (Sigla)'] = alunos_para_prever['Previsão (Situação)'].map(reverse_mapping_dict)

# 7. Exibir o resultado no terminal
print("Previsões realizadas:")
print(alunos_para_prever[['Previsão (Situação)', 'Previsão (Sigla)']])

# 8. Salvar o resultado em um novo CSV
alunos_para_prever.to_csv(os.path.join(base_dir, 'resultados_previsoes_xgboost.csv'), index=False, encoding='latin1', sep=';')

print("Previsões salvas em 'resultados_previsoes_xgboost.csv'")
