#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para gerar o relatório final por aluno, incluindo previsões,
probabilidades e Top 3 features SHAP mais influentes.
(Versão v2 - Robusta à execução da pasta raiz do projeto)
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import sys
import os
import re

# --- Configuração de Caminhos (Assumindo execução da raiz do projeto) ---
def get_project_paths_from_root():
    """Determina os caminhos importantes assumindo execução da raiz."""
    paths = {}
    try:
        paths["project_root"] = os.path.abspath(os.getcwd()) # Assume execução da raiz
        paths["xgboost_dir"] = os.path.join(paths["project_root"], "XGBoost")
        paths["shap_analysis_dir"] = os.path.join(paths["xgboost_dir"], "shap_analysis")
        print(f"Diretório raiz detectado: {paths['project_root']}")
        print(f"Pasta XGBoost esperada: {paths['xgboost_dir']}")
        print(f"Pasta SHAP esperada: {paths['shap_analysis_dir']}")
        return paths
    except Exception as e:
        print(f"Erro ao determinar caminhos a partir da raiz: {e}")
        return None

paths = get_project_paths_from_root()
if paths is None:
    sys.exit(1)

# Adicionar raiz ao sys.path para importar mapeamento (se necessário)
if paths["project_root"] not in sys.path:
    sys.path.insert(0, paths["project_root"])

# --- Importações e Carregamentos --- 
try:
    # Tenta importar diretamente, pois a raiz já deve estar no path
    from mapeamento_classes import mapping_dict, reverse_mapping_dict
    print("Importação de mapeamento OK.")
except ImportError:
    print(f"Erro: Não foi possível importar de 'mapeamento_classes.py' localizado em {paths['project_root']}")
    sys.exit(1)

def load_model(xgboost_dir, project_root):
    """Carrega o modelo XGBoost de locais conhecidos relativos à raiz."""
    # Caminho 1: Dentro da pasta XGBoost
    model_path_xgb = os.path.join(xgboost_dir, "modelo_xgboost_otimizado_mapeado.pkl")
    # Caminho 2: Dentro de output/modelos na raiz
    model_path_output = os.path.join(project_root, "output", "modelos", "modelo_xgboost_otimizado_mapeado.pkl")
    
    modelo_path = None
    print(f"Procurando modelo em: {model_path_xgb}")
    if os.path.exists(model_path_xgb):
        modelo_path = model_path_xgb
    else:
        print(f"Modelo não encontrado. Procurando em: {model_path_output}")
        if os.path.exists(model_path_output):
            modelo_path = model_path_output
        else:
            print(f"Erro: Modelo não encontrado em {model_path_xgb} ou {model_path_output}")
            return None
            
    try:
        modelo = joblib.load(modelo_path)
        print(f"Modelo carregado de: {modelo_path}")
        return modelo
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return None

def load_shap_results(shap_dir):
    """Carrega os valores SHAP e informações de mapeamento da pasta SHAP."""
    shap_values_path = os.path.join(shap_dir, "shap_values.npy")
    mapping_info_path = os.path.join(shap_dir, "class_mapping_info.txt")
    
    print(f"Procurando arquivos SHAP em: {shap_dir}")
    if not os.path.exists(shap_values_path) or not os.path.exists(mapping_info_path):
        print(f"Erro: Arquivos SHAP ('shap_values.npy' ou 'class_mapping_info.txt') não encontrados em {shap_dir}")
        print("Certifique-se de executar 'calculate_shap_values_relative.py' na pasta 'XGBoost/shap_analysis/' primeiro.")
        return None, None
        
    try:
        shap_values = np.load(shap_values_path)
        with open(mapping_info_path, "r") as f:
            content = f.read()
            match = re.search(r"Nomes das classes finais: \[(.*?)\]", content)
            if match:
                class_names_str = match.group(1)
                final_class_names = [name.strip().strip("'") for name in class_names_str.split(",")]
            else:
                raise ValueError("Não foi possível encontrar 'Nomes das classes finais' no arquivo de mapeamento.")
        print(f"Valores SHAP carregados: {shap_values.shape}")
        print(f"Nomes das classes finais (SHAP): {final_class_names}")
        return shap_values, final_class_names
    except Exception as e:
        print(f"Erro ao carregar resultados SHAP: {e}")
        return None, None

def prepare_data_for_prediction(csv_path):
    """Carrega e prepara os dados, mantendo colunas originais."""
    try:
        df_orig = pd.read_csv(csv_path, encoding="latin1", sep=";")
        print(f"Dados originais carregados: {df_orig.shape}")
        df = df_orig.copy()
    except Exception as e:
        print(f"Erro ao carregar dados originais de {csv_path}: {e}")
        return None, None

    # Aplicar mesmos filtros e mapeamentos da preparação SHAP
    classe_counts = df["Situação (código)"].value_counts()
    classes_para_remover = classe_counts[classe_counts < 10].index
    df = df[~df["Situação (código)"].isin(classes_para_remover)]
    agrupamento_classes = {3: 2}
    df["Situação (código)"] = df["Situação (código)"].replace(agrupamento_classes)
    df["Situacao_Mapeada_Inicial"] = df["Situação (código)"].map(mapping_dict)
    df.dropna(subset=["Situacao_Mapeada_Inicial"], inplace=True)
    df["Situacao_Mapeada_Inicial"] = df["Situacao_Mapeada_Inicial"].astype(int)

    # Manter apenas as linhas que foram usadas no treinamento/SHAP
    df_processed = df.copy()

    # Preparar X para o modelo
    X = df.drop(columns=["Situação (código)", "Situacao_Mapeada_Inicial"])
    colunas_para_remover = ["Matrícula", "Nome", "Curso", "Renda", "Sexo",
                            "Bairro", "Cidade", "Turma Atual",
                            "Pend. Financ.", "Situação", "Descrição"]
    X = X.drop(columns=[col for col in colunas_para_remover if col in X.columns], errors='ignore')
    
    print(f"Dados X preparados para previsão: {X.shape}")
    return df_processed, X # Retorna df com linhas filtradas e X para previsão

def get_top_shap_features(shap_values_instance, feature_names, n=3):
    """Extrai as top N features SHAP positivas e negativas."""
    feature_shap = list(zip(feature_names, shap_values_instance))
    feature_shap.sort(key=lambda x: x[1], reverse=True)
    top_pos = [f"{feat} ({val:.2f})" for feat, val in feature_shap[:n] if val > 0]
    top_neg = [f"{feat} ({val:.2f})" for feat, val in feature_shap[-n:][::-1] if val < 0]
    while len(top_pos) < n: top_pos.append("-")
    while len(top_neg) < n: top_neg.append("-")
    return top_pos, top_neg

# --- Execução Principal --- 
if __name__ == "__main__":
    print("Gerando Relatório Final por Aluno (v2)...")

    # 1. Carregar Modelo
    model = load_model(paths["xgboost_dir"], paths["project_root"])
    if model is None: sys.exit(1)

    # 2. Carregar Resultados SHAP
    shap_values, final_class_names_shap = load_shap_results(paths["shap_analysis_dir"])
    if shap_values is None: sys.exit(1)

    # 3. Preparar Dados
    csv_path = os.path.join(paths["project_root"], "planilha_final.csv")
    df_processed, X = prepare_data_for_prediction(csv_path)
    if df_processed is None: sys.exit(1)
    
    # Garantir que as colunas de X estejam na ordem esperada pelo modelo
    if hasattr(model, 'feature_names_in_'):
        try:
            X = X[model.feature_names_in_]
            print("Colunas de X reordenadas conforme o modelo.")
        except Exception as e:
            print(f"Erro ao reordenar colunas de X: {e}. Verifique as features.")
            sys.exit(1)
    else:
        print("Aviso: Modelo não tem 'feature_names_in_'. Assumindo que a ordem das colunas está correta.")

    feature_names = list(X.columns)

    # 4. Fazer Previsões
    print("Realizando previsões...")
    y_pred_indices = model.predict(X)
    y_pred_probs = model.predict_proba(X)
    y_pred_names = [final_class_names_shap[idx] for idx in y_pred_indices]
    pred_probabilities = [y_pred_probs[i, idx] for i, idx in enumerate(y_pred_indices)]

    # 5. Montar DataFrame de Resultados
    print("Montando DataFrame de resultados...")
    results_df = df_processed[["Matrícula", "Nome", "Curso", "Situação"]].copy()
    results_df["Situacao_Prevista"] = y_pred_names
    results_df["Probabilidade_Prevista"] = [f"{p:.2%}" for p in pred_probabilities]

    # 6. Extrair Features SHAP (Top 3)
    print("Extraindo Top 3 features SHAP por aluno...")
    top_pos_all = []
    top_neg_all = []
    for i in range(len(X)):
        pred_class_idx = y_pred_indices[i]
        shap_instance_pred_class = shap_values[i, :, pred_class_idx]
        top_pos, top_neg = get_top_shap_features(shap_instance_pred_class, feature_names, n=3)
        top_pos_all.append(top_pos)
        top_neg_all.append(top_neg)

    results_df[["Top1_Pos", "Top2_Pos", "Top3_Pos"]] = pd.DataFrame(top_pos_all, index=results_df.index)
    results_df[["Top1_Neg", "Top2_Neg", "Top3_Neg"]] = pd.DataFrame(top_neg_all, index=results_df.index)

    # 7. Salvar Relatório CSV na pasta raiz
    output_csv_path = os.path.join(paths["project_root"], "relatorio_alunos_predicoes_shap.csv")
    try:
        results_df.to_csv(output_csv_path, index=False, sep=';', encoding='latin1')
        print(f"Relatório final salvo com sucesso em: {output_csv_path}")
    except Exception as e:
        print(f"Erro ao salvar o relatório CSV: {e}")

    print("Processo concluído.")
