import pandas as pd
import joblib

# 1. Carregar o modelo treinado
modelo = joblib.load('modelo_logistico.pkl')

# 2. Criar novo aluno (com mesmas colunas usadas no treino)
novo_aluno = pd.DataFrame([{
    'Módulo atual': 1,
    'Faltas Consecutivas': 0,
    'Histórico de reprovações': 3,
    'Histórico de Recuperação': 4,
    'Historico de Reprovado por Falta (disciplinas)': 0,
    'Idade': 19,
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
