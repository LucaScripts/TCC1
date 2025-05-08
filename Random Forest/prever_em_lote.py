import pandas as pd
import joblib

# 1. Carregar o modelo treinado
modelo = joblib.load('modelo_random_forest.pkl')  # Carrega o modelo Random Forest treinado

# 2. Carregar os dados dos alunos para prever
alunos_para_prever = pd.read_csv('alunos_para_prever.csv', encoding='latin1', sep=';')  # Substitua pelo caminho do arquivo

# 3. Garantir que as colunas correspondam às usadas no treinamento
# (Certifique-se de que o arquivo CSV já tenha as colunas corretas)

# 4. Fazer as previsões
predicoes = modelo.predict(alunos_para_prever)

# 5. Adicionar as previsões ao DataFrame
alunos_para_prever['Previsão (Situação)'] = predicoes

# 6. Adicionar as siglas correspondentes às previsões
codigo_para_sigla = {
    0: 'CAC', 1: 'CAI', 2: 'CAN', 3: 'CAU', 4: 'ES',
    5: 'FO', 6: 'LAC', 7: 'LFI', 8: 'LFR', 9: 'MT',
    10: 'NC', 11: 'NF', 12: 'TF', 13: 'TR'
}
alunos_para_prever['Previsão (Sigla)'] = alunos_para_prever['Previsão (Situação)'].map(codigo_para_sigla)

# 7. Exibir os resultados no terminal
print("Previsões realizadas:")
print(alunos_para_prever[['Previsão (Situação)', 'Previsão (Sigla)']])  # Exibe as previsões e as siglas no terminal

# 8. Salvar os resultados em um novo arquivo CSV
alunos_para_prever.to_csv('resultados_previsoes.csv', index=False, encoding='latin1', sep=';')

print("Previsões salvas em 'resultados_previsoes.csv'")