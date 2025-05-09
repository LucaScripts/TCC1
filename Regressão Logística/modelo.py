import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import joblib

# Carregar dados
df = pd.read_csv('planilha_final.csv', encoding='latin1', sep=';')

# Remover classes pequenas
classe_counts = df['Situação (código)'].value_counts()
classes_para_remover = classe_counts[classe_counts < 10].index
df = df[~df['Situação (código)'].isin(classes_para_remover)]

# Features e alvo
X = df.drop(columns=['Situação (código)', 'Curso', 'Bairro', 'Cidade'])
y = df['Situação (código)']

# Separar treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Balancear com SMOTE
smote = SMOTE(k_neighbors=1, random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Treinar modelo de Regressão Logística
modelo = LogisticRegression(max_iter=1000, random_state=42)
modelo.fit(X_train_bal, y_train_bal)

# Avaliação
y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))

# Salvar modelo treinado
joblib.dump(modelo, 'modelo_logistico.pkl')
print('Modelo salvo como modelo_logistico.pkl')
