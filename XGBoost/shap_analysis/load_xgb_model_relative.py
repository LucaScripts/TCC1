#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para carregar o modelo XGBoost otimizado para análise SHAP (versão com caminhos relativos)
"""
import joblib
import os
import sys
import pandas as pd

def carregar_modelo():
    print("Carregando o modelo XGBoost otimizado (caminhos relativos)...")
    
    # Diretório base do script (shap_analysis)
    script_dir = os.path.dirname(__file__)
    # Diretório do modelo (XGBoost)
    model_dir = os.path.abspath(os.path.join(script_dir, ".."))
    # Diretório raiz do projeto (TCC1)
    project_root = os.path.abspath(os.path.join(model_dir, ".."))
    
    # Caminhos relativos para procurar o modelo
    # Prioridade 1: Na pasta XGBoost
    modelo_path_xgb = os.path.join(model_dir, "modelo_xgboost_otimizado_mapeado.pkl")
    # Prioridade 2: Na pasta output/modelos na raiz do projeto
    modelo_path_output = os.path.join(project_root, "output", "modelos", "modelo_xgboost_otimizado_mapeado.pkl")
    
    modelo_path = None
    if os.path.exists(modelo_path_xgb):
        modelo_path = modelo_path_xgb
        print(f"Modelo encontrado em: {modelo_path_xgb}")
    elif os.path.exists(modelo_path_output):
        modelo_path = modelo_path_output
        print(f"Modelo encontrado em: {modelo_path_output}")
    else:
        print(f"Erro: Modelo não encontrado em {modelo_path_xgb} ou {modelo_path_output}")
        return None
    
    # Carregar o modelo
    try:
        modelo = joblib.load(modelo_path)
        print("Modelo carregado com sucesso!")
        
        # Verificar se é um modelo XGBoost
        if hasattr(modelo, 'feature_importances_'):
            print("Verificação de modelo: OK - Possui feature_importances_")
        else:
            print("Aviso: O modelo não parece ter o atributo feature_importances_")
        
        # Salvar informações do modelo na pasta atual (shap_analysis)
        output_dir = script_dir
        info_path = os.path.join(output_dir, "model_info.txt")
        with open(info_path, "w") as f:
            f.write(f"Modelo carregado de: {modelo_path}\n")
            f.write(f"Tipo do modelo: {type(modelo)}\n")
            if hasattr(modelo, 'feature_importances_'):
                f.write(f"Número de features: {len(modelo.feature_importances_)}\n")
            if hasattr(modelo, 'classes_'):
                f.write(f"Classes: {modelo.classes_}\n")
                f.write(f"Número de classes: {len(modelo.classes_)}\n")
        print(f"Informações do modelo salvas em {info_path}")
        
        return modelo
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return None

if __name__ == "__main__":
    modelo = carregar_modelo()
    if modelo is not None:
        print("Modelo carregado e pronto para análise SHAP.")
    else:
        print("Falha ao carregar o modelo. Verifique os erros acima.")
