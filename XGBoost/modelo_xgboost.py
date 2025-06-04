# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import time
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib
import sys
import os # Importar os

# --- Ajuste de Caminho para Importação --- 
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
from mapeamento_classes import mapping_dict, reverse_mapping_dict as initial_reverse_mapping

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

# 4. Aplicar o mapeamento inicial (pode ter gaps)
df["Situacao_Mapeada_Inicial"] = df["Situação (código)"].map(mapping_dict)

# Verificar e tratar NaNs
if df["Situacao_Mapeada_Inicial"].isnull().any():
    print("Aviso: Códigos não mapeados!")
    print(df[df["Situacao_Mapeada_Inicial"].isnull()]["Situação (código)"].unique())
    df.dropna(subset=["Situacao_Mapeada_Inicial"], inplace=True)
df["Situacao_Mapeada_Inicial"] = df["Situacao_Mapeada_Inicial"].astype(int)

# 5. Separar X e y inicial
X = df.drop(columns=["Situação (código)", "Situacao_Mapeada_Inicial"])
y_inicial = df["Situacao_Mapeada_Inicial"]

# 6. Remover colunas não usadas
colunas_para_remover = ["Matrícula", "Nome", "Curso", "Renda", "Sexo", "Bairro", "Cidade", "Turma Atual", "Pend. Financ.", "Situação", "Descrição"]
X = X.drop(columns=[col for col in colunas_para_remover if col in X.columns])

# 7. Separar treino e teste (usando y inicial)
X_train, X_test, y_train_inicial, y_test_inicial = train_test_split(X, y_inicial, test_size=0.3, random_state=42, stratify=y_inicial)

# --- Mapeamento Final Contíguo (0 a N-1) --- 
final_le = LabelEncoder()
y_combined = pd.concat([y_train_inicial, y_test_inicial])
final_le.fit(y_combined)
y_train = final_le.transform(y_train_inicial)
y_test = final_le.transform(y_test_inicial)
final_classes_indices = final_le.classes_
final_class_names = [initial_reverse_mapping[idx] for idx in final_classes_indices]
num_final_classes = len(final_classes_indices)
print(f"Labels iniciais presentes: {final_classes_indices}")
print(f"Labels finais contíguos: {np.unique(y_train)}")
print(f"Nomes das classes finais: {final_class_names}")
print(f"Número de classes finais: {num_final_classes}")
# --- Fim do Mapeamento Final ---

# 8. Timestamp
timestamp = time.strftime("%Y%m%d-%H%M%S")

# 9. Plotar distribuição antes do balanceamento
plt.figure(figsize=(10, 6))
pd.Series(y_train).map(dict(zip(range(num_final_classes), final_class_names))).value_counts().plot(kind="bar", color="skyblue")
plt.title("Distribuição das Classes Finais (Antes do SMOTETomek) - XGBoost")
plt.xlabel("Classes (Sigla)")
plt.ylabel("Quantidade")
plt.xticks(rotation=45)
plt.tight_layout()
nome_arquivo_antes = os.path.join(graficos_dir, f"xgb_dist_antes_smote_{timestamp}.png")
plt.savefig(nome_arquivo_antes)
plt.close()
print(f"Gráfico antes salvo em: {nome_arquivo_antes}")

# 10. Aplicar SMOTETomek
min_samples = pd.Series(y_train).value_counts().min()
k_neighbors = min(5, max(1, min_samples - 1))
smt = SMOTETomek(smote=SMOTE(k_neighbors=k_neighbors, random_state=42), random_state=42)
print(f"Usando k_neighbors={k_neighbors} para SMOTE dentro do SMOTETomek")
X_train_bal, y_train_bal = smt.fit_resample(X_train, y_train)

# 11. Plotar distribuição depois do balanceamento
plt.figure(figsize=(10, 6))
pd.Series(y_train_bal).map(dict(zip(range(num_final_classes), final_class_names))).value_counts().plot(kind="bar", color="lightgreen")
plt.title("Distribuição das Classes Finais (Após SMOTETomek) - XGBoost")
plt.xlabel("Classes (Sigla)")
plt.ylabel("Quantidade")
plt.xticks(rotation=45)
plt.tight_layout()
nome_arquivo_depois = os.path.join(graficos_dir, f"xgb_dist_depois_smote_{timestamp}.png")
plt.savefig(nome_arquivo_depois)
plt.close()
print(f"Gráfico depois salvo em: {nome_arquivo_depois}")

# 13. Treinar modelo inicial
modelo_inicial = XGBClassifier(objective="multi:softprob",
                               num_class=num_final_classes,
                               use_label_encoder=False,
                               eval_metric="mlogloss",
                               random_state=42)
modelo_inicial.fit(X_train_bal, y_train_bal)

# 14. Avaliar modelo inicial
y_pred_inicial = modelo_inicial.predict(X_test)
print("\nRelatório de Classificação (Modelo Inicial - XGBoost Final Mapeado):")
print(classification_report(y_test, y_pred_inicial, labels=range(num_final_classes), target_names=final_class_names, zero_division=0))

# 15. Salvar modelo inicial
modelo_inicial_path = os.path.join(modelos_dir, "modelo_xgboost_inicial_mapeado.pkl")
joblib.dump(modelo_inicial, modelo_inicial_path)
print(f"Modelo inicial salvo em: {modelo_inicial_path}")

# 16. Tuning - RandomizedSearchCV
param_dist = {
    "n_estimators": [100, 200, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 10, 15, 20],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.1, 0.2, 0.5]
}

print("\nIniciando RandomizedSearchCV...")
random_search = RandomizedSearchCV(
    XGBClassifier(objective="multi:softprob",
                  num_class=num_final_classes,
                  use_label_encoder=False,
                  eval_metric="mlogloss",
                  random_state=42),
    param_distributions=param_dist,
    n_iter=30,
    cv=5,
    scoring="f1_macro",
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_bal, y_train_bal)

print("\nMelhores parâmetros encontrados:", random_search.best_params_)
modelo_otimizado = random_search.best_estimator_

# 17. Avaliação modelo otimizado
y_pred_otimizado = modelo_otimizado.predict(X_test)
print("\nRelatório de Classificação (Modelo Otimizado - XGBoost Final Mapeado):")
print(classification_report(y_test, y_pred_otimizado, labels=range(num_final_classes), target_names=final_class_names, zero_division=0))

# 18. Salvar modelo otimizado
modelo_otimizado_path = os.path.join(modelos_dir, "modelo_xgboost_otimizado_mapeado.pkl")
joblib.dump(modelo_otimizado, modelo_otimizado_path)
print(f"Modelo otimizado salvo em: {modelo_otimizado_path}")

# 19. Importância das Features
importances = modelo_otimizado.feature_importances_
indices = importances.argsort()[::-1]
plt.figure(figsize=(12, 6))
plt.title("Importância das Features - XGBoost Final Mapeado")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
imp_features_path = os.path.join(graficos_dir, "xgb_importancia_features_mapeado.png")
plt.savefig(imp_features_path)
plt.close()
print(f"Gráfico da importância das features salvo em: {imp_features_path}")

# 20. Validação Cruzada
print("\nCalculando F1-Score médio da validação cruzada...")
scores = cross_val_score(modelo_otimizado, X_train_bal, y_train_bal, cv=5, scoring="f1_macro")
print(f"F1-Score Médio (Validação Cruzada - XGBoost Final Mapeado): {scores.mean():.4f}")

