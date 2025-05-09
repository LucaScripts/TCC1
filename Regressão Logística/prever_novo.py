import pandas as pd
import joblib

# 1. Carregar o modelo treinado
modelo = joblib.load('modelo_logistico.pkl')

# 2. Criar novo aluno (com mesmas colunas usadas no treino)
novo_aluno = pd.DataFrame([{
    'Módulo atual': 2,
    'Faltas Consecutivas': 3,
    'Idade': 16,
    'Sexo (código)': 1, 
    'Pend. Acad.': 0,
    'Possui Pendência Financeira': 0,
    'Bolsista': 1,
    'Antecipou Parcela': 0
}])

# 3. Fazer a previsão
predicao = modelo.predict(novo_aluno)

# 4. Mostrar o resultado
print(f'Previsão da Situação (código): {predicao[0]}')
