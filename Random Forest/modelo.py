import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from datetime import datetime
import matplotlib.pyplot as plt
import joblib

# 1. Carregar os dados
df = pd.read_csv('planilha_final.csv', encoding='latin1', sep=';')

# 2. Remover classes com poucos exemplos
classe_counts = df['Situação (código)'].value_counts()
classes_para_remover = classe_counts[classe_counts < 20].index
df = df[~df['Situação (código)'].isin(classes_para_remover)]

# 3. Agrupar CAU (3) para CAN (2)
agrupamento_classes = {3: 2}
df['Situação (código)'] = df['Situação (código)'].replace(agrupamento_classes)

# 4. Atualizar descrição das classes
descricao_classes = {
    0: 'CAC', 1: 'CAI', 2: 'CAN', 4: 'ES', 5: 'FO',
    6: 'LAC', 7: 'LFI', 8: 'LFR', 9: 'MT', 10: 'NC', 12: 'TF'
}

# 5. Separar X e y
X = df.drop(columns=['Situação (código)'])
y = df['Situação (código)']

# 6. Remover colunas categóricas
colunas_para_remover = ['Matrícula', 'Nome', 'Curso', 'Renda', 'Sexo', 
                        'Bairro', 'Cidade', 'Turma Atual', 
                        'Pend. Financ.', 'Situação', 'Descrição']
X = X.drop(columns=colunas_para_remover)

# 7. Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=42, 
                                                    stratify=y)

# 8. Timestamp
timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

# 9. Plotar gráfico de distribuição antes do balanceamento
plt.figure(figsize=(10,6))
y_train.value_counts().rename(index=descricao_classes).plot(kind='bar', color='skyblue')
plt.title('Distribuição das Classes (Antes do SMOTETomek)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'grafico_classes_antes_smote_{timestamp}.png')
plt.close()

# 10. Codificar y
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# 11. Aplicar SMOTETomek
smote_tomek = SMOTETomek(random_state=42)
X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train, y_train_encoded)

# 12. Plotar gráfico depois do balanceamento
plt.figure(figsize=(10,6))
pd.Series(y_train_bal).value_counts().rename(index=dict(enumerate(label_encoder.classes_))).plot(kind='bar', color='lightgreen')
plt.title('Distribuição das Classes (Após SMOTETomek)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'grafico_classes_depois_smote_{timestamp}.png')
plt.close()

# 13. Treinar modelo inicial
modelo_inicial = RandomForestClassifier(random_state=42, class_weight='balanced')
modelo_inicial.fit(X_train_bal, y_train_bal)

# 14. Avaliação inicial
y_pred_inicial = modelo_inicial.predict(X_test)
print("Relatório de Classificação (Modelo Inicial):")
print(classification_report(y_test_encoded, y_pred_inicial))

# 15. Salvar modelo inicial
joblib.dump(modelo_inicial, 'modelo_random_forest_inicial.pkl')
print("Modelo inicial salvo como 'modelo_random_forest_inicial.pkl'")

# 16. Tuning - RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'),
                                   param_distributions=param_dist,
                                   n_iter=30,  # agora 30 buscas aleatórias
                                   cv=5,  # validação cruzada 5 folds
                                   scoring='f1_macro',
                                   verbose=2,
                                   random_state=42,
                                   n_jobs=-1)
random_search.fit(X_train_bal, y_train_bal)

print("Melhores parâmetros encontrados:", random_search.best_params_)
modelo_otimizado = random_search.best_estimator_

# 17. Avaliação modelo otimizado
y_pred_otimizado = modelo_otimizado.predict(X_test)
print("Relatório de Classificação (Modelo Otimizado):")
print(classification_report(y_test_encoded, y_pred_otimizado))

# 18. Salvar modelo otimizado
joblib.dump(modelo_otimizado, 'modelo_random_forest_otimizado.pkl')
print("Modelo otimizado salvo como 'modelo_random_forest_otimizado.pkl'")

# 19. Importância das Features
importances = modelo_otimizado.feature_importances_
indices = importances.argsort()[::-1]
plt.figure(figsize=(12,6))
plt.title('Importância das Features')
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
plt.savefig('importancia_features.png')
plt.close()
print("Gráfico da importância das features salvo como 'importancia_features.png'")

# 20. Validação Cruzada
scores = cross_val_score(modelo_otimizado, X_train_bal, y_train_bal, cv=5, scoring='f1_macro')
print(f"F1-Score Médio (Validação Cruzada): {scores.mean():.4f}")

