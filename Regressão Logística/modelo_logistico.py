import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use('Agg')  # Modo sem interface gráfica
import matplotlib.pyplot as plt
import time
import joblib

# 1. Carregar os dados
df = pd.read_csv('planilha_final.csv', encoding='latin1', sep=';')

# 2. Remover classes com menos de 10 exemplos
classe_counts = df['Situação (código)'].value_counts()
classes_para_remover = classe_counts[classe_counts < 10].index
df = df[~df['Situação (código)'].isin(classes_para_remover)]

# 3. Separar features (X) e alvo (y)
X = df.drop(columns=['Situação (código)'])
y = df['Situação (código)']

# 4. Remover colunas categóricas
colunas_para_remover = ['Matrícula','Nome','Curso','Histórico de reprovações','Renda','Sexo','Bairro','Cidade','Turma Atual','Pend. Financ.','Situação','Descrição']
X = X.drop(columns=colunas_para_remover)

# 5. Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 6. Plotar a distribuição antes do balanceamento
# Criar uma coluna combinando a sigla e o código
df['Sigla_Codigo'] = df['Situação'] + ' (' + df['Situação (código)'].astype(str) + ')'

# Atualizar os rótulos do eixo X no gráfico
y_train.value_counts().rename(index=lambda x: df[df['Situação (código)'] == x]['Sigla_Codigo'].iloc[0]).plot(
    kind='bar', title='Distribuição das Classes (Antes do SMOTE)'
)
plt.xlabel('Classes (Sigla e Código)')
plt.ylabel('Quantidade')
plt.tight_layout()
timestamp = time.strftime("%Y%m%d-%H%M%S")
nome_arquivo = f'grafico_classes_logistico_{timestamp}.png'
plt.savefig(nome_arquivo)
plt.close()
print(f"Gráfico salvo como: {nome_arquivo}")

# 7. Aplicar SMOTE
smote = SMOTE(k_neighbors=1, random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# 8. Treinar o modelo
modelo = LogisticRegression(max_iter=200, random_state=42)
modelo.fit(X_train_bal, y_train_bal)

# 9. Avaliar
y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))

# 10. Salvar o modelo
joblib.dump(modelo, 'modelo_logistico.pkl')
print("Modelo salvo como modelo_logistico.pkl")
