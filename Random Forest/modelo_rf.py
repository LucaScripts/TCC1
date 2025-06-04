# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import sys
import os # Importar os

# --- Ajuste de Caminho para Importação --- 
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
from mapeamento_classes import mapping_dict, reverse_mapping_dict, class_names_ordered

# --- Caminhos de Arquivos e Pastas --- 
csv_path = os.path.join(parent_dir, "planilha_final.csv")
output_dir = os.path.join(parent_dir, "output")
graficos_dir = os.path.join(output_dir, "graficos")
modelos_dir = os.path.join(output_dir, "modelos")

# Criar pastas de output
os.makedirs(graficos_dir, exist_ok=True)
os.makedirs(modelos_dir, exist_ok=True)
# --- Fim dos Caminhos ---

# 1. Carregar os dados
print(f"Carregando dados de: {csv_path}")
df = pd.read_csv(csv_path, encoding="latin1", sep=";")

# 2. Remover classes com poucos exemplos
classe_counts = df["Situação (código)"].value_counts()
classes_para_remover = classe_counts[classe_counts < 10].index
df = df[~df["Situação (código)"].isin(classes_para_remover)]

# 3. Agrupar CAU (3) para CAN (2)
agrupamento_classes = {3: 2}
df["Situação (código)"] = df["Situação (código)"].replace(agrupamento_classes)

# 4. Aplicar o mapeamento padrão
df["Situacao_Mapeada"] = df["Situação (código)"].map(mapping_dict)

# Verificar e tratar NaNs
if df["Situacao_Mapeada"].isnull().any():
    print("Aviso: Códigos não mapeados!")
    print(df[df["Situacao_Mapeada"].isnull()]["Situação (código)"].unique())
    df.dropna(subset=["Situacao_Mapeada"], inplace=True)
df["Situacao_Mapeada"] = df["Situacao_Mapeada"].astype(int)

# 5. Separar X e y
X = df.drop(columns=["Situação (código)", "Situacao_Mapeada"])
y = df["Situacao_Mapeada"]

# 6. Remover colunas não usadas
colunas_para_remover = ["Matrícula", "Nome", "Curso", "Renda", "Sexo", "Bairro", "Cidade", "Turma Atual", "Pend. Financ.", "Situação", "Descrição"]
X = X.drop(columns=[col for col in colunas_para_remover if col in X.columns])

# 7. Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Obter labels únicos presentes
labels_presentes = sorted(np.unique(np.concatenate((y_train, y_test))))
target_names_presentes = [reverse_mapping_dict[label] for label in labels_presentes]
print(f"Labels presentes nos dados: {labels_presentes}")
print(f"Nomes das classes presentes: {target_names_presentes}")

# 8. Timestamp
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

# 9. Plotar distribuição antes do balanceamento
plt.figure(figsize=(10, 6))
y_train.map(reverse_mapping_dict).value_counts().plot(kind="bar", color="skyblue")
plt.title("Distribuição das Classes (Antes do SMOTETomek) - RF")
plt.xlabel("Classes (Sigla)")
plt.ylabel("Quantidade")
plt.xticks(rotation=45)
plt.tight_layout()
nome_arquivo_antes = os.path.join(graficos_dir, f"rf_dist_antes_smote_{timestamp}.png")
plt.savefig(nome_arquivo_antes)
plt.close()
print(f"Gráfico antes do SMOTE salvo em: {nome_arquivo_antes}")

# 11. Aplicar SMOTETomek
min_samples = y_train.value_counts().min()
k_neighbors = min(5, max(1, min_samples - 1))
smt = SMOTETomek(smote=SMOTE(k_neighbors=k_neighbors, random_state=42), random_state=42)
print(f"Usando k_neighbors={k_neighbors} para SMOTE dentro do SMOTETomek")
X_train_bal, y_train_bal = smt.fit_resample(X_train, y_train)

# 12. Plotar distribuição depois do balanceamento
plt.figure(figsize=(10, 6))
pd.Series(y_train_bal).map(reverse_mapping_dict).value_counts().plot(kind="bar", color="lightgreen")
plt.title("Distribuição das Classes (Após SMOTETomek) - RF")
plt.xlabel("Classes (Sigla)")
plt.ylabel("Quantidade")
plt.xticks(rotation=45)
plt.tight_layout()
nome_arquivo_depois = os.path.join(graficos_dir, f"rf_dist_depois_smote_{timestamp}.png")
plt.savefig(nome_arquivo_depois)
plt.close()
print(f"Gráfico após SMOTE salvo em: {nome_arquivo_depois}")

# 13. Treinar modelo inicial
modelo_inicial = RandomForestClassifier(random_state=42, class_weight="balanced")
modelo_inicial.fit(X_train_bal, y_train_bal)

# 14. Avaliação inicial
y_pred_inicial = modelo_inicial.predict(X_test)
print("\nRelatório de Classificação (Modelo Inicial - RF Mapeado):")
print(classification_report(y_test, y_pred_inicial, labels=range(len(class_names_ordered)), target_names=class_names_ordered, zero_division=0))

# 15. Salvar modelo inicial
modelo_inicial_path = os.path.join(modelos_dir, "modelo_rf_inicial_mapeado.pkl")
joblib.dump(modelo_inicial, modelo_inicial_path)
print(f"Modelo inicial salvo em: {modelo_inicial_path}")

# 16. Tuning - RandomizedSearchCV
param_dist = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None]
}

print("\nIniciando RandomizedSearchCV...")
random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42, class_weight="balanced"),
                                   param_distributions=param_dist,
                                   n_iter=30,
                                   cv=5,
                                   scoring="f1_macro",
                                   verbose=1,
                                   random_state=42,
                                   n_jobs=-1)
random_search.fit(X_train_bal, y_train_bal)

print("\nMelhores parâmetros encontrados:", random_search.best_params_)
modelo_otimizado = random_search.best_estimator_

# 17. Avaliação modelo otimizado
y_pred_otimizado = modelo_otimizado.predict(X_test)
print("\nRelatório de Classificação (Modelo Otimizado - RF Mapeado):")
print(classification_report(y_test, y_pred_otimizado, labels=range(len(class_names_ordered)), target_names=class_names_ordered, zero_division=0))

# 18. Salvar modelo otimizado
modelo_otimizado_path = os.path.join(modelos_dir, "modelo_rf_otimizado_mapeado.pkl")
joblib.dump(modelo_otimizado, modelo_otimizado_path)
print(f"Modelo otimizado salvo em: {modelo_otimizado_path}")

# 19. Importância das Features
importances = modelo_otimizado.feature_importances_
indices = importances.argsort()[::-1]
plt.figure(figsize=(12, 6))
plt.title("Importância das Features - RF Mapeado")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
imp_features_path = os.path.join(graficos_dir, "rf_importancia_features_mapeado.png")
plt.savefig(imp_features_path)
plt.close()
print(f"Gráfico da importância das features salvo em: {imp_features_path}")

# 20. Validação Cruzada
print("\nCalculando F1-Score médio da validação cruzada...")
scores = cross_val_score(modelo_otimizado, X_train_bal, y_train_bal, cv=5, scoring="f1_macro")
print(f"F1-Score Médio (Validação Cruzada - RF Mapeado): {scores.mean():.4f}")

