import pandas as pd
import joblib

# 1. Carregar o modelo treinado
modelo = joblib.load('modelo_random_forest_inicial.pkl')  # Carrega o modelo Random Forest treinado

# 2. Criar um novo exemplo de aluno (com as mesmas colunas usadas no treino!)
novo_aluno = pd.DataFrame([{
    'Módulo atual': 1,
    'Faltas Consecutivas': 3,
    'Histórico de reprovações': 10,
    'Histórico de Recuperação': 0,
    'Historico de Reprovado por Falta (disciplinas)': 0,
    'Idade': 23,
    'Sexo (código)': 0, 
    'Pend. Acad.': 0,
    'Possui Pendência Financeira': 0,
    'Bolsista': 0,
    'Antecipou Parcela': 0
}])

# 3. Fazer a previsão
predicao = modelo.predict(novo_aluno)

# 4. Mostrar o resultado
print(f'Previsão da Situação (código): {predicao[0]}')
