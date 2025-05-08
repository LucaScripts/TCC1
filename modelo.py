import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use('Agg')  # Usar backend que não abre janela

import matplotlib.pyplot as plt  # <-- Faltava isso!
import pandas as pd

# Carregar os dados
df = pd.read_csv('planilha_final.csv', encoding='latin1', sep=';')  # Usa separador correto!

# Separar features (X) e alvo (y)
X = df.drop('Situação (código)', axis=1)
y = df['Situação (código)']

# Visualizar distribuição inicial
print("Distribuição das classes (antes do SMOTE):")
print(y.value_counts())

# Plotar gráfico de distribuição antes
y.value_counts().plot(kind='bar', title='Distribuição das Classes (Antes do SMOTE)')
plt.savefig('grafico_classes.png')

# Dividir entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicar SMOTE no conjunto de treino
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Visualizar distribuição depois do SMOTE
print("\nDistribuição das classes (depois do SMOTE):")
print(pd.Series(y_train_bal).value_counts())

# Plotar gráfico de distribuição depois
pd.Series(y_train_bal).value_counts().plot(kind='bar', title='Distribuição das Classes (Depois do SMOTE)')
plt.savefig('grafico_classes.png')

# Treinar modelo
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train_bal, y_train_bal)

# Fazer previsões
y_pred = modelo.predict(X_test)

# Avaliar desempenho
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred))
