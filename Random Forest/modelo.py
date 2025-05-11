import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from datetime import datetime
import matplotlib.pyplot as plt
import joblib

# Carregar os dados
df = pd.read_csv('planilha_final.csv', encoding='latin1', sep=';')

# Remover classes com menos de 20 exemplos
classe_counts = df['Situação (código)'].value_counts()
classes_para_remover = classe_counts[classe_counts < 20].index
df = df[~df['Situação (código)'].isin(classes_para_remover)]

# Verificar distribuição
print("Distribuição das classes após a remoção de classes com poucos exemplos:")
print(df['Situação (código)'].value_counts())

# Dicionário de descrição das classes
descricao_classes = {
    0: 'CAC', 1: 'CAI', 2: 'CAN', 3: 'CAU', 4: 'ES',
    5: 'FO', 6: 'LAC', 7: 'LFI', 8: 'LFR', 9: 'MT',
    10: 'NC', 11: 'NF', 12: 'TF', 13: 'TR'
}

# Separar features e variável alvo
X = df.drop(columns=['Situação (código)'])
y = df['Situação (código)']

# Remover colunas categóricas
colunas_para_remover = ['Matrícula', 'Nome', 'Curso', 'Renda', 'Sexo', 'Bairro', 
                        'Cidade', 'Turma Atual', 'Pend. Financ.', 'Situação', 'Descrição']
X = X.drop(columns=colunas_para_remover)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Timestamp
timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

# Gráfico de distribuição antes do SMOTE
plt.figure(figsize=(12, 8))
y_train.value_counts().rename(index=descricao_classes).plot(kind='bar', color='skyblue')
plt.title("Distribuição das Classes Antes do SMOTE", fontsize=16)
plt.xlabel("Classes", fontsize=14)
plt.ylabel("Número de Exemplos", fontsize=14)
plt.xticks(rotation=45, fontsize=10)
plt.tight_layout()
nome_arquivo_antes = f"grafico_classes_antes_smote_{timestamp}.png"
plt.savefig(nome_arquivo_antes)
print(f"Gráfico salvo: '{nome_arquivo_antes}'")

# Label Encoding
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print("Classes antes do SMOTE (codificadas):", label_encoder.classes_)

# Aplicar SMOTE
smote = SMOTE(k_neighbors=1, random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train_encoded)

# Gráfico de distribuição depois do SMOTE
y_train_bal_decodificado = label_encoder.inverse_transform(y_train_bal)
plt.figure(figsize=(12, 8))
pd.Series(y_train_bal_decodificado).value_counts().rename(index=descricao_classes).plot(kind='bar', color='lightgreen')
plt.title("Distribuição das Classes Depois do SMOTE", fontsize=16)
plt.xlabel("Classes", fontsize=14)
plt.ylabel("Número de Exemplos", fontsize=14)
plt.xticks(rotation=45, fontsize=10)
plt.tight_layout()
nome_arquivo_depois = f"grafico_classes_depois_smote_{timestamp}.png"
plt.savefig(nome_arquivo_depois)
print(f"Gráfico salvo: '{nome_arquivo_depois}'")

# Treinar modelo inicial
modelo = RandomForestClassifier(random_state=42, class_weight='balanced')
modelo.fit(X_train_bal, y_train_bal)

# Avaliar modelo inicial
y_pred = modelo.predict(X_test)
print("Relatório de classificação (modelo inicial):")
print(classification_report(y_test_encoded, y_pred))

# Salvar o modelo inicial
joblib.dump(modelo, 'modelo_random_forest_inicial.pkl')
print("Modelo inicial salvo como 'modelo_random_forest_inicial.pkl'")

# Ajuste de Hiperparâmetros
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_distributions=param_dist,
    n_iter=10, cv=3, scoring='f1_macro', random_state=42, verbose=2
)
random_search.fit(X_train_bal, y_train_bal)

print("Melhores parâmetros encontrados:", random_search.best_params_)
modelo_otimizado = random_search.best_estimator_

# Avaliar modelo otimizado
y_pred_otimizado = modelo_otimizado.predict(X_test)
print("Relatório de classificação (modelo otimizado):")
print(classification_report(y_test_encoded, y_pred_otimizado))

# Salvar o modelo otimizado
joblib.dump(modelo_otimizado, 'modelo_random_forest_otimizado.pkl')
print("Modelo otimizado salvo como 'modelo_random_forest_otimizado.pkl'")

# Importância das Features
importances = modelo_otimizado.feature_importances_
indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
feature_names = X_train.columns

plt.figure(figsize=(10, 6))
plt.title("Importância das Features")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
plt.tight_layout()
plt.savefig('importancia_features.png')
print("Gráfico de importância das features salvo como 'importancia_features.png'")

# Validação Cruzada
scores = cross_val_score(modelo_otimizado, X_train_bal, y_train_bal, cv=5, scoring='f1_macro')
print("F1-score médio (validação cruzada):", scores.mean())

