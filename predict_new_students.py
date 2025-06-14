#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para gerar previsões (situação e probabilidade) para novos alunos
usando o modelo XGBoost treinado e os dados pré-processados.
(Versão v3 - Correção de sintaxe na f-string)
"""
import pandas as pd
import joblib
import os
import sys

# --- Configuração de Caminhos (Assumindo execução da raiz do projeto) ---
def get_project_paths_from_root():
    """Determina os caminhos importantes assumindo execução da raiz."""
    paths = {}
    try:
        paths["project_root"] = os.path.abspath(os.getcwd()) # Assume execução da raiz
        paths["xgboost_dir"] = os.path.join(paths["project_root"], "XGBoost")
        paths["shap_analysis_dir"] = os.path.join(paths["xgboost_dir"], "shap_analysis") # Para mapeamento
        # CORREÇÃO: Usar aspas simples dentro da f-string
        print(f"Diretório raiz detectado: {paths['project_root']}") 
        return paths
    except Exception as e:
        print(f"Erro ao determinar caminhos a partir da raiz: {e}")
        return None

paths = get_project_paths_from_root()
if paths is None:
    sys.exit(1)

# Adicionar raiz ao sys.path para importar mapeamento
if paths["project_root"] not in sys.path:
    sys.path.insert(0, paths["project_root"])

# --- Importações e Carregamentos --- 
try:
    from mapeamento_classes import reverse_mapping_dict
    print("Importação de reverse_mapping_dict OK.")
except ImportError:
    print(f"Erro: Não foi possível importar reverse_mapping_dict de 'mapeamento_classes.py' localizado em {paths['project_root']}")
    sys.exit(1)

def load_model(xgboost_dir, project_root):
    """Carrega o modelo XGBoost de locais conhecidos relativos à raiz."""
    model_path_xgb = os.path.join(xgboost_dir, "modelo_xgboost_otimizado_mapeado.pkl")
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

# --- Execução Principal --- 
if __name__ == "__main__":
    print("Iniciando geração de previsões para novos alunos (v3)...")

    # 1. Carregar Modelo XGBoost
    model = load_model(paths["xgboost_dir"], paths["project_root"])
    if model is None: sys.exit(1)

    # 2. Carregar Dados Pré-processados e IDs
    X_new_path = os.path.join(paths["project_root"], "X_novos_alunos_preprocessado.csv")
    ids_path = os.path.join(paths["project_root"], "ids_novos_alunos.csv")
    
    if not os.path.exists(X_new_path) or not os.path.exists(ids_path):
        print(f"Erro: Arquivos pré-processados (\n"
              f"    'X_novos_alunos_preprocessado.csv' ou 'ids_novos_alunos.csv'\n"
              f"    ) não encontrados em {paths['project_root']}")
        print("Certifique-se de executar 'preprocess_new_students_v3.py' primeiro.")
        sys.exit(1)
        
    try:
        X_new = pd.read_csv(X_new_path, sep=";", encoding="latin1")
        df_ids = pd.read_csv(ids_path, sep=";", encoding="latin1")
        print(f"Dados pré-processados X carregados: {X_new.shape}")
        print(f"IDs carregados: {df_ids.shape}")
        if len(X_new) != len(df_ids):
            print("Erro: Número de linhas em X_novos_alunos e ids_novos_alunos não correspondem!")
            sys.exit(1)
        df_ids.index = X_new.index
    except Exception as e:
        print(f"Erro ao carregar arquivos pré-processados: {e}")
        sys.exit(1)

    # 3. Verificar consistência das features
    if hasattr(model, 'feature_names_in_'):
        model_features = list(model.feature_names_in_)
        if list(X_new.columns) != model_features:
            print("Erro: Colunas nos dados pré-processados não correspondem às esperadas pelo modelo!")
            print(f"Esperado: {model_features}")
            print(f"Encontrado: {list(X_new.columns)}")
            try:
                print("Tentando reordenar colunas...")
                X_new = X_new[model_features]
                print("Colunas reordenadas com sucesso.")
            except Exception as e_reorder:
                print(f"Erro ao reordenar colunas: {e_reorder}")
                sys.exit(1)
    else:
        print("Aviso: Modelo não possui 'feature_names_in_'. Verificação de ordem de features pulada.")

    # 4. Gerar Previsões e Probabilidades
    print("Gerando previsões...")
    try:
        y_pred_indices = model.predict(X_new)
        y_pred_probs = model.predict_proba(X_new)
        print("Previsões geradas.")
    except Exception as e:
        print(f"Erro ao gerar previsões: {e}")
        sys.exit(1)

    # 5. Mapear Índices para Nomes e Obter Probabilidade da Classe Prevista
    try:
        print("Usando reverse_mapping_dict para nomes das classes.")
        predicted_class_names = [reverse_mapping_dict.get(idx, f"Índice_{idx}_Desconhecido") for idx in y_pred_indices]
        predicted_probabilities = [y_pred_probs[i, idx] for i, idx in enumerate(y_pred_indices)]
        print("Nomes e probabilidades das classes previstas extraídos.")
    except KeyError as e:
         print(f"Erro de Mapeamento: Índice previsto {e} não encontrado em reverse_mapping_dict.")
         print("Verifique se reverse_mapping_dict em mapeamento_classes.py está completo.")
         sys.exit(1)
    except Exception as e:
        print(f"Erro ao mapear previsões ou extrair probabilidades: {e}")
        sys.exit(1)

    # 6. Montar DataFrame Final
    print("Montando DataFrame final...")
    df_final_predictions = df_ids.copy()
    df_final_predictions['Situacao_Prevista'] = predicted_class_names
    df_final_predictions['Probabilidade_Prevista'] = [f"{p:.2%}" for p in predicted_probabilities]

    # 7. Salvar Relatório CSV
    output_csv_path = os.path.join(paths["project_root"], "previsoes_novos_alunos.csv")
    try:
        df_final_predictions.to_csv(output_csv_path, index=False, sep=";", encoding="latin1")
        print(f"Relatório de previsões para novos alunos salvo com sucesso em: {output_csv_path}")
    except Exception as e:
        print(f"Erro ao salvar o relatório CSV de previsões: {e}")

    print("Processo de previsão concluído.")
