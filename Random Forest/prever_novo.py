import pandas as pd
import joblib

# 1. Carregar o modelo treinado
modelo = joblib.load('modelo_random_forest.pkl')

# 2. Criar um novo exemplo de aluno (com as mesmas colunas usadas no treino!)
novo_aluno = pd.DataFrame([{
    'Módulo atual': 2,
    'Faltas Consecutivas': 0,
    'Idade': 25,
    'Sexo': 1,
    'Pend. Acad.': 0,
    'Possui Pendência Financeira': 2,
    'Bolsista': 0,
    'Antecipou Parcela': 0
}])

# 3. Fazer a previsão
predicao = modelo.predict(novo_aluno)

# 4. Mostrar o resultado
print(f'Previsão da Situação (código): {predicao[0]}')
