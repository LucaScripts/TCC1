import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

import matplotlib
matplotlib.use('Agg')  # Configura o matplotlib para modo sem interface gráfica
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Carregar os dados
df = pd.read_csv('planilha_final.csv', encoding='latin1', sep=';')

# Separar features e variável alvo
X = df.drop(columns=['Situação (código)'])
y = df['Situação (código)']

# REMOVE colunas categóricas ANTES do split
colunas_para_remover = ['Curso', 'Bairro', 'Cidade']
X = X.drop(columns=colunas_para_remover)

# Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Exibir distribuição original
print('Distribuição das classes (antes do SMOTE):')
print(y_train.value_counts())

# Plotar distribuição das classes
y_train.value_counts().plot(kind='bar', title='Distribuição das Classes (Antes do SMOTE)')
plt.xlabel('Classes')
plt.ylabel('Quantidade')
plt.tight_layout()
plt.savefig('grafico_classes.png')  # Salva o gráfico
plt.close()  # Fecha o gráfico para não travar

# Aplicar SMOTE
smote = SMOTE(k_neighbors=1, random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Treinar modelo (exemplo: Random Forest)
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train_bal, y_train_bal)

# Avaliar modelo
y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))
