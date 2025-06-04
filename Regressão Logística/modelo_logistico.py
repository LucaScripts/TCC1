# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import joblib
import sys
import os # Importar os para manipulação de caminhos e pastas

# --- Ajuste de Caminho para Importação --- 
# Adiciona o diretório pai (tcc_mvp1) ao path para encontrar mapeamento_classes.py
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
from mapeamento_classes import mapping_dict, reverse_mapping_dict, class_names_ordered

# --- Caminhos de Arquivos e Pastas --- 
# Caminho relativo para o CSV na pasta pai
csv_path = os.path.join(parent_dir, "planilha_final.csv")

# Caminhos para as pastas de output na pasta pai
output_dir = os.path.join(parent_dir, "output")
graficos_dir = os.path.join(output_dir, "graficos")
modelos_dir = os.path.join(output_dir, "modelos")

# Criar pastas de output se não existirem
os.makedirs(graficos_dir, exist_ok=True)
os.makedirs(modelos_dir, exist_ok=True)
# --- Fim dos Caminhos ---

# 1. Carregar os dados
print(f"Carregando dados de: {csv_path}")
df = pd.read_csv(csv_path, encoding="latin1", sep=";")

# 2. Remover classes com menos de 10 exemplos (NF, TR)
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

# 5. Separar features (X) e alvo (y)
X = df.drop(columns=["Situação (código)", "Situacao_Mapeada"])
y = df["Situacao_Mapeada"]

# 6. Remover colunas não usadas
colunas_para_remover = ["Matrícula", "Nome", "Curso", "Renda", "Sexo", "Bairro", "Cidade", "Turma Atual", "Pend. Financ.", "Situação", "Descrição", "Sigla_Codigo"]
X = X.drop(columns=[col for col in colunas_para_remover if col in X.columns])

# Guardar nomes das features para o gráfico de importância
feature_names = X.columns

# 7. Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Obter labels únicos presentes
labels_presentes = sorted(np.unique(np.concatenate((y_train, y_test))))
target_names_presentes = [reverse_mapping_dict[label] for label in labels_presentes]
print(f"Labels presentes nos dados: {labels_presentes}")
print(f"Nomes das classes presentes: {target_names_presentes}")

# 8. Plotar distribuição antes do balanceamento
plt.figure(figsize=(10, 6))
y_train.map(reverse_mapping_dict).value_counts().plot(kind="bar", title="Distribuição das Classes (Antes do SMOTE) - Logístico")
plt.xlabel("Classes (Sigla)")
plt.ylabel("Quantidade")
plt.xticks(rotation=45)
plt.tight_layout()
timestamp = time.strftime("%Y%m%d-%H%M%S")
nome_arquivo = os.path.join(graficos_dir, f"logistico_dist_antes_smote_{timestamp}.png")
plt.savefig(nome_arquivo)
plt.close()
print(f"Gráfico antes do SMOTE salvo em: {nome_arquivo}")

# 9. Aplicar SMOTE
min_samples = y_train.value_counts().min()
k_neighbors = min(5, max(1, min_samples - 1))
print(f"Usando k_neighbors={k_neighbors} para SMOTE")
smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# 10. Plotar distribuição depois do balanceamento
plt.figure(figsize=(10, 6))
pd.Series(y_train_bal).map(reverse_mapping_dict).value_counts().plot(kind="bar", title="Distribuição das Classes (Após SMOTE) - Logístico")
plt.xlabel("Classes (Sigla)")
plt.ylabel("Quantidade")
plt.xticks(rotation=45)
plt.tight_layout()
nome_arquivo_depois = os.path.join(graficos_dir, f"logistico_dist_depois_smote_{timestamp}.png")
plt.savefig(nome_arquivo_depois)
plt.close()
print(f"Gráfico após SMOTE salvo em: {nome_arquivo_depois}")

# 11. Treinar o modelo
modelo = LogisticRegression(max_iter=3000, random_state=42, solver="saga", multi_class="multinomial", class_weight="balanced")
modelo.fit(X_train_bal, y_train_bal)

# 12. Avaliar
y_pred = modelo.predict(X_test)
print("\nRelatório de Classificação (Regressão Logística - Mapeamento Padrão):")
print(classification_report(y_test, y_pred, labels=range(len(class_names_ordered)), target_names=class_names_ordered, zero_division=0))

# --- Adicionado: Gráfico de Importância das Features (baseado em coeficientes) ---
# Como é multiclasse, pegamos a média da magnitude absoluta dos coeficientes por feature
if hasattr(modelo, 'coef_'):
    # Calcular a importância média da magnitude absoluta dos coeficientes
    importance = np.mean(np.abs(modelo.coef_), axis=0)
    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(12, 6))
    plt.title("Importância das Features - Regressão Logística (Média Coefs Abs)")
    plt.bar(range(X_train.shape[1]), importance[indices])
    plt.xticks(range(X_train.shape[1]), feature_names[indices], rotation=90)
    plt.ylabel("Importância Média (Magnitude Absoluta Coeficiente)")
    plt.tight_layout()
    imp_features_path = os.path.join(graficos_dir, "logistico_importancia_features.png")
    plt.savefig(imp_features_path)
    plt.close()
    print(f"Gráfico da importância das features salvo em: {imp_features_path}")
else:
    print("Modelo não possui coeficientes para análise de importância.")
# --- Fim da adição ---

# 13. Salvar o modelo
modelo_path = os.path.join(modelos_dir, "modelo_logistico_mapeado.pkl")
joblib.dump(modelo, modelo_path)
print(f"\nModelo salvo em: {modelo_path}")

