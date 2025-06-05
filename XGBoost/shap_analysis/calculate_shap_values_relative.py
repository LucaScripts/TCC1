#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para calcular os valores SHAP para o modelo XGBoost otimizado (versão com caminhos relativos).
"""
import shap
import pandas as pd
import joblib
import os
import numpy as np
import sys

# Adicionar o diretório atual ao path para importar o script de carregamento
sys.path.append(os.path.dirname(__file__))

try:
    # Importar a função do script relativo
    from load_xgb_model_relative import carregar_modelo
except ImportError:
    print("Erro: Não foi possível importar 'load_xgb_model_relative'. Certifique-se de que ambos os scripts estão na mesma pasta (shap_analysis).")
    sys.exit(1)

def calcular_shap():
    print("Iniciando cálculo dos valores SHAP (caminhos relativos)...")
    
    # 1. Definir diretório de dados (pasta atual do script)
    data_dir = os.path.dirname(__file__)
    X_path = os.path.join(data_dir, "X_prepared.csv")
    
    if not os.path.exists(X_path):
        print(f"Erro: Arquivo de dados preparados não encontrado em {X_path}")
        return
        
    try:
        X = pd.read_csv(X_path)
        print(f"Dados X carregados: {X.shape[0]} linhas, {X.shape[1]} colunas")
    except Exception as e:
        print(f"Erro ao carregar X_prepared.csv: {e}")
        return

    # 2. Carregar modelo treinado usando a função do script relativo
    modelo = carregar_modelo()
    if modelo is None:
        print("Erro: Falha ao carregar o modelo. Abortando cálculo SHAP.")
        return
        
    # Verificar consistência de features
    if hasattr(modelo, 'feature_names_in_') and list(modelo.feature_names_in_) != list(X.columns):
         print("Erro: Nomes/ordem das features no modelo não correspondem aos dados carregados!")
         print("Features no modelo:", list(modelo.feature_names_in_))
         print("Features nos dados:", list(X.columns))
         try:
             print("Tentando reordenar colunas dos dados...")
             X = X[modelo.feature_names_in_]
             print("Colunas reordenadas com sucesso.")
         except Exception as e:
             print(f"Erro ao reordenar colunas: {e}. Abortando.")
             return
    elif not hasattr(modelo, 'feature_names_in_'):
         print("Aviso: Modelo não possui 'feature_names_in_'. Verificação de features pulada.")
         if hasattr(modelo, 'n_features_in_') and modelo.n_features_in_ != X.shape[1]:
             print(f"Erro: Número de features no modelo ({modelo.n_features_in_}) não corresponde aos dados ({X.shape[1]}). Abortando.")
             return
         elif not hasattr(modelo, 'n_features_in_'):
              print("Aviso: Modelo também não possui 'n_features_in_'. Não foi possível verificar features.")

    # 3. Inicializar o TreeExplainer
    try:
        print("Inicializando SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(modelo, feature_names=X.columns)
        print("Explainer inicializado.")
    except Exception as e:
        print(f"Erro ao inicializar o TreeExplainer: {e}")
        return

    # 4. Calcular os valores SHAP
    print("Calculando valores SHAP (pode levar alguns minutos)...")
    try:
        shap_values = explainer.shap_values(X, check_additivity=False)
        print("Valores SHAP calculados com sucesso.")
        if isinstance(shap_values, list):
            print(f"Formato SHAP values: Lista com {len(shap_values)} arrays (um por classe)")
            print(f"Shape do array SHAP para a primeira classe: {shap_values[0].shape}")
        elif isinstance(shap_values, np.ndarray):
             print(f"Formato SHAP values: {type(shap_values)}, Shape: {shap_values.shape}")
        else:
            print(f"Formato SHAP values inesperado: {type(shap_values)}")
            
    except Exception as e:
        print(f"Erro durante o cálculo dos valores SHAP: {e}")
        return

    # 5. Salvar os valores SHAP e base values na pasta atual
    shap_values_path = os.path.join(data_dir, "shap_values.npy")
    base_values_path = os.path.join(data_dir, "shap_base_values.npy")
    
    try:
        print("Salvando valores SHAP...")
        np.save(shap_values_path, shap_values)
        print(f"Valores SHAP salvos em {shap_values_path}")
        
        if hasattr(explainer, 'expected_value'):
             print("Salvando base values (expected_value)...")
             np.save(base_values_path, explainer.expected_value)
             print(f"Base values salvos em {base_values_path}")
        else:
             print("Aviso: Explainer não possui expected_value.")
             
    except Exception as e:
        print(f"Erro ao salvar os resultados SHAP: {e}")

    print("Cálculo e salvamento dos valores SHAP concluídos.")

if __name__ == "__main__":
    calcular_shap()
