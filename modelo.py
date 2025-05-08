import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

import matplotlib
matplotlib.use('Agg')  # Configura o matplotlib para modo sem interface gráfica
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Carregar os dados
df = pd.read_csv('planilha_final.csv', encoding='latin1', sep=';')  # Lê o arquivo CSV com separador ';' e codificação 'latin1'

# Remove classes com menos de 10 exemplos
classe_counts = df['Situação (código)'].value_counts()  # Conta a quantidade de exemplos por classe
classes_para_remover = classe_counts[classe_counts < 10].index  # Identifica classes com menos de 10 exemplos
df = df[~df['Situação (código)'].isin(classes_para_remover)]  # Remove as classes com poucos exemplos

# Separar features e variável alvo
X = df.drop(columns=['Situação (código)'])  # Remove a coluna alvo do conjunto de features
y = df['Situação (código)']  # Define a variável alvo

# REMOVE colunas categóricas ANTES do split
colunas_para_remover = ['Curso', 'Bairro', 'Cidade']  # Define as colunas categóricas a serem removidas
X = X.drop(columns=colunas_para_remover)  # Remove as colunas categóricas do conjunto de features

# Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  
# Divide os dados em treino (70%) e teste (30%) com uma semente fixa para reprodutibilidade

# Exibir distribuição original
print('Distribuição das classes (antes do SMOTE):')
print(y_train.value_counts())  # Mostra a distribuição das classes no conjunto de treino

# Plotar distribuição das classes
y_train.value_counts().plot(kind='bar', title='Distribuição das Classes (Antes do SMOTE)')  
# Cria um gráfico de barras para visualizar a distribuição das classes
plt.xlabel('Classes')  # Define o rótulo do eixo X
plt.ylabel('Quantidade')  # Define o rótulo do eixo Y
plt.tight_layout()  # Ajusta o layout para evitar sobreposição
plt.savefig('grafico_classes.png')  # Salva o gráfico em um arquivo
plt.close()  # Fecha o gráfico para liberar memória

# Aplicar SMOTE
smote = SMOTE(k_neighbors=1, random_state=42)  # Instancia o SMOTE com 1 vizinho e semente fixa
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)  
# Aplica o SMOTE para balancear as classes no conjunto de treino

# Treinar modelo (exemplo: Random Forest)
modelo = RandomForestClassifier(random_state=42)  # Instancia o modelo Random Forest com semente fixa
modelo.fit(X_train_bal, y_train_bal)  # Treina o modelo com os dados balanceados

# Avaliar modelo
y_pred = modelo.predict(X_test)  # Faz previsões no conjunto de teste
print(classification_report(y_test, y_pred))  
# Exibe o relatório de classificação com métricas como precisão, recall e F1-score
