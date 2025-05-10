import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import time
from sklearn.preprocessing import LabelEncoder

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib

# Carregar os dados
df = pd.read_csv('planilha_final.csv', encoding='latin1', sep=';')

# Remove classes com menos de 10 exemplos
classe_counts = df['Situação (código)'].value_counts()
classes_para_remover = classe_counts[classe_counts < 10].index
df = df[~df['Situação (código)'].isin(classes_para_remover)]

# Separar features e variável alvo
X = df.drop(columns=['Situação (código)'])
y = df['Situação (código)']

# Remover colunas categóricas
colunas_para_remover = ['Matrícula', 'Nome', 'Curso', 'Renda', 'Sexo', 
                        'Bairro', 'Cidade', 'Turma Atual', 'Pend. Financ.', 'Situação', 'Descrição']
X = X.drop(columns=colunas_para_remover)

# Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Dicionário para mapear códigos às siglas
codigo_para_sigla = {
    0: 'CAC', 1: 'CAI', 2: 'CAN', 3: 'CAU', 4: 'ES',
    5: 'FO', 6: 'LAC', 7: 'LFI', 8: 'LFR', 9: 'MT',
    10: 'NC', 11: 'NF', 12: 'TF', 13: 'TR'
}

# Exibir distribuição original
print('Distribuição das classes (antes do SMOTE):')
print(y_train.value_counts())

# Criar gráfico da distribuição
codigo_e_sigla = {codigo: f"{codigo} - {sigla}" for codigo, sigla in codigo_para_sigla.items()}
y_train_rotulado = y_train.replace(codigo_e_sigla)

y_train_rotulado.value_counts().plot(kind='bar', title='Distribuição das Classes (Antes do SMOTE)')
plt.xlabel('Classes (Código + Sigla)')
plt.ylabel('Quantidade')
plt.tight_layout()

timestamp = time.strftime("%Y%m%d-%H%M%S")
nome_arquivo = f'grafico_classes_xgboost_{timestamp}.png'
plt.savefig(nome_arquivo)
plt.close()

print(f"Gráfico salvo como: {nome_arquivo}")

# Aplicar SMOTE
smote = SMOTE(k_neighbors=1, random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Reindexar as classes
le = LabelEncoder()
y_train_bal = le.fit_transform(y_train_bal)
y_test = le.transform(y_test)  # Muito importante transformar o y_test também!

# Treinar o modelo XGBoost
modelo = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
modelo.fit(X_train_bal, y_train_bal)

# Avaliar modelo
y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))

# Salvar o modelo
joblib.dump(modelo, 'modelo_xgboost.pkl')
print("Modelo salvo como modelo_xgboost.pkl")
