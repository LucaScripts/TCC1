#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script UNIFICADO para gerar previsões (situação, probabilidade e Top 3 fatores SHAP) para novos alunos.
(Versão v5 - Correção para array SHAP 3D (amostras, features, classes))
Lê 'alunos_para_prever.csv', pré-processa, carrega o modelo, prevê, calcula SHAP e salva
o resultado final em 'previsoes_novos_alunos_unificado.csv'.
"""
import pandas as pd
import numpy as np
import joblib
import os
import sys
import shap

# --- Configuração de Caminhos (Assumindo execução da raiz do projeto) ---
def get_project_paths_from_root():
    """Determina os caminhos importantes assumindo execução da raiz."""
    paths = {}
    try:
        paths["project_root"] = os.path.abspath(os.getcwd()) # Assume execução da raiz
        paths["xgboost_dir"] = os.path.join(paths["project_root"], "XGBoost")
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

# --- Importações e Constantes --- 
try:
    from mapeamento_classes import reverse_mapping_dict
    print("Importação de reverse_mapping_dict OK.")
except ImportError:
    print(f"Erro: Não foi possível importar reverse_mapping_dict de 'mapeamento_classes.py' localizado em {paths['project_root']}")
    sys.exit(1)

# Ordem Esperada das Features (Baseada nos scripts anteriores)
EXPECTED_FEATURE_ORDER = [
    'Módulo atual', 
    'Faltas Consecutivas', 
    'Histórico de reprovações', 
    'Histórico de Recuperação', 
    'Historico de Reprovado por Falta (disciplinas)', 
    'Idade', 
    'Sexo (código)', 
    'Pend. Acad.', 
    'Possui Pendência Financeira', 
    'Bolsista', 
    'Antecipou Parcela'
]

# --- Funções Auxiliares --- 
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

def calculate_shap_values(model, X):
    """Calcula os valores SHAP para as previsões."""
    try:
        print("Calculando valores SHAP...")
        # Criar o explicador SHAP para modelos baseados em árvores
        explainer = shap.TreeExplainer(model)
        
        # Calcular valores SHAP
        shap_values = explainer.shap_values(X)
        
        # Verificar o formato dos valores SHAP
        if isinstance(shap_values, list):
            print(f"Valores SHAP calculados com sucesso. Formato: lista com {len(shap_values)} elementos.")
            for i, sv in enumerate(shap_values):
                print(f"  - Classe {i}: shape {sv.shape}")
        else:
            print(f"Valores SHAP calculados com sucesso. Formato: array com shape {shap_values.shape}")
        
        return shap_values, explainer.expected_value
    except Exception as e:
        print(f"Erro ao calcular valores SHAP: {e}")
        return None, None

def get_top_features_from_shap_values(shap_values_for_sample, feature_names, top_n=3):
    """
    Identifica as top N features com maior impacto positivo e negativo para uma amostra.
    
    Args:
        shap_values_for_sample: Valores SHAP para uma amostra específica (array 1D)
        feature_names: Lista com os nomes das features
        top_n: Número de features a retornar (padrão: 3)
        
    Returns:
        top_positive: Lista de tuplas (feature, valor) com as features de maior impacto positivo
        top_negative: Lista de tuplas (feature, valor) com as features de maior impacto negativo
    """
    try:
        # Verificar se o número de valores SHAP corresponde ao número de features
        if len(shap_values_for_sample) != len(feature_names):
            print(f"Aviso: Número de valores SHAP ({len(shap_values_for_sample)}) não corresponde ao número de features ({len(feature_names)})")
            # Usar o menor dos dois para evitar erros de índice
            min_len = min(len(shap_values_for_sample), len(feature_names))
            shap_values_for_sample = shap_values_for_sample[:min_len]
            feature_names = feature_names[:min_len]
        
        # Criar pares (feature_name, shap_value)
        feature_shap_pairs = list(zip(feature_names, shap_values_for_sample))
        
        # Ordenar por valor SHAP (do maior para o menor)
        sorted_pairs = sorted(feature_shap_pairs, key=lambda x: float(x[1]), reverse=True)
        
        # Obter as top N features positivas
        top_positive = [(f, float(v)) for f, v in sorted_pairs if float(v) > 0][:top_n]
        # Se não houver features positivas suficientes, preencher com vazios
        while len(top_positive) < top_n:
            top_positive.append(("", 0.0))
            
        # Ordenar por valor SHAP (do menor para o maior)
        sorted_pairs_neg = sorted(feature_shap_pairs, key=lambda x: float(x[1]))
        top_negative = [(f, float(v)) for f, v in sorted_pairs_neg if float(v) < 0][:top_n]
        # Se não houver features negativas suficientes, preencher com vazios
        while len(top_negative) < top_n:
            top_negative.append(("", 0.0))
            
        return top_positive, top_negative
    except Exception as e:
        print(f"Erro ao identificar top features SHAP: {e}")
        # Retornar listas vazias em caso de erro
        empty_list = [("", 0.0) for _ in range(top_n)]
        return empty_list, empty_list

def format_shap_feature(feature_name, value, X_row):
    """
    Formata uma feature SHAP para exibição, incluindo o valor da feature.
    
    Args:
        feature_name: Nome da feature
        value: Valor SHAP
        X_row: Linha do DataFrame com os valores das features
        
    Returns:
        String formatada com nome da feature e seu valor
    """
    try:
        # Se a feature estiver vazia (caso de preenchimento), retornar string vazia
        if not feature_name:
            return ""
            
        feature_value = X_row[feature_name]
        if feature_name in ['Possui Pendência Financeira', 'Pend. Acad.', 'Bolsista', 'Antecipou Parcela']:
            # Features binárias
            status = "Sim" if feature_value == 1 else "Não"
            return f"{feature_name} ({status})"
        elif feature_name == 'Sexo (código)':
            gender = "Masculino" if feature_value == 1 else "Feminino"
            return f"{feature_name} ({gender})"
        else:
            # Features numéricas
            if feature_name == 'Idade':
                return f"{feature_name} ({int(feature_value)} anos)"
            elif feature_name == 'Módulo atual':
                return f"{feature_name} ({int(feature_value)})"
            elif feature_name == 'Faltas Consecutivas':
                return f"{feature_name} ({int(feature_value)})"
            else:
                return f"{feature_name} ({feature_value})"
    except Exception as e:
        print(f"Erro ao formatar feature SHAP: {e}")
        return feature_name if feature_name else ""

# --- Execução Principal Unificada --- 
if __name__ == "__main__":
    print("Iniciando processo UNIFICADO de previsão para novos alunos (v5 com SHAP - correção para array 3D)...")
    
    # == ETAPA 1: Ler e Pré-processar CSV de Novos Alunos ==
    print("\n--- Etapa 1: Lendo e Pré-processando Novos Alunos ---")
    feature_order = EXPECTED_FEATURE_ORDER
    print(f"Ordem esperada das features: {feature_order}")

    input_csv_path = os.path.join(paths["project_root"], "alunos_para_prever.csv")
    if not os.path.exists(input_csv_path):
        print(f"Erro: Arquivo 'alunos_para_prever.csv' não encontrado em {paths['project_root']}")
        sys.exit(1)
        
    try:
        df_new = pd.read_csv(input_csv_path, encoding="latin1", sep=";")
        print(f"Dados de novos alunos carregados (latin1): {df_new.shape}")
    except UnicodeDecodeError:
        try:
            df_new = pd.read_csv(input_csv_path, encoding="utf-8", sep=";")
            print(f"Dados de novos alunos carregados (utf-8): {df_new.shape}")
        except Exception as e:
            print(f"Erro ao carregar {input_csv_path} com latin1 e utf-8: {e}")
            sys.exit(1)
    except Exception as e:
         print(f"Erro ao carregar {input_csv_path}: {e}")
         sys.exit(1)
        
    # Guardar IDs
    id_cols_to_keep = ['Matrícula', 'Nome', 'Curso']
    actual_id_cols = [col for col in id_cols_to_keep if col in df_new.columns]
    if not actual_id_cols:
        print("Aviso: Nenhuma coluna de identificação (Matrícula, Nome, Curso) encontrada.")
        df_ids = pd.DataFrame(index=df_new.index)
    else:
         df_ids = df_new[actual_id_cols].copy()

    # Verificar colunas necessárias
    missing_cols = [col for col in feature_order if col not in df_new.columns]
    if missing_cols:
        print(f"Erro: Colunas necessárias ausentes no arquivo 'alunos_para_prever.csv': {missing_cols}")
        print(f"Colunas encontradas: {list(df_new.columns)}")
        sys.exit(1)
        
    # Selecionar e reordenar features
    try:
        X_new = df_new[feature_order].copy()
        print(f"Features selecionadas e reordenadas: {X_new.shape}")
    except Exception as e:
        print(f"Erro ao selecionar/reordenar features: {e}")
        sys.exit(1)
        
    # Tratar NaNs
    if X_new.isnull().values.any():
        print("Aviso: Existem valores nulos (NaN) nas features selecionadas.")
        print("Aplicando preenchimento de NaNs com 0 (Atenção: revise se esta é a melhor estratégia)...")
        X_new.fillna(0, inplace=True)
        if X_new.isnull().values.any():
             print("Erro: Falha ao preencher NaNs.")
             sys.exit(1)
        else:
             print("NaNs preenchidos com 0.")
    print("Pré-processamento concluído.")

    # == ETAPA 2: Carregar Modelo ==
    print("\n--- Etapa 2: Carregando Modelo XGBoost ---")
    model = load_model(paths["xgboost_dir"], paths["project_root"])
    if model is None: sys.exit(1)

    # Verificar consistência das features (opcional)
    if hasattr(model, 'feature_names_in_'):
        model_features = list(model.feature_names_in_)
        if list(X_new.columns) != model_features:
            print("Erro: Colunas nos dados pré-processados não correspondem às esperadas pelo modelo!")
            # ... (poderia tentar reordenar ou parar)
            print("Abortando.")
            sys.exit(1)
    else:
        print("Aviso: Modelo não possui 'feature_names_in_'. Verificação de ordem de features pulada.")

    # == ETAPA 3: Gerar Previsões ==
    print("\n--- Etapa 3: Gerando Previsões ---")
    try:
        y_pred_indices = model.predict(X_new)
        y_pred_probs = model.predict_proba(X_new)
        print("Previsões geradas.")
        
        # Verificar o número de classes no modelo
        n_classes = y_pred_probs.shape[1]
        print(f"Número de classes no modelo: {n_classes}")
    except Exception as e:
        print(f"Erro ao gerar previsões: {e}")
        sys.exit(1)

    # == ETAPA 4: Calcular Valores SHAP ==
    print("\n--- Etapa 4: Calculando Valores SHAP ---")
    shap_values, expected_values = calculate_shap_values(model, X_new)
    if shap_values is None:
        print("Aviso: Não foi possível calcular valores SHAP. Continuando sem análise SHAP.")
        include_shap = False
    else:
        include_shap = True
        print("Valores SHAP calculados com sucesso.")

    # == ETAPA 5: Mapear Resultados e Montar DataFrame Final ==
    print("\n--- Etapa 5: Mapeando Resultados e Montando Relatório ---")
    try:
        print("Usando reverse_mapping_dict para nomes das classes.")
        predicted_class_names = [reverse_mapping_dict.get(idx, f"Índice_{idx}_Desconhecido") for idx in y_pred_indices]
        predicted_probabilities = [y_pred_probs[i, idx] for i, idx in enumerate(y_pred_indices)]
        print("Nomes e probabilidades das classes previstas extraídos.")
    except KeyError as e:
         print(f"Erro de Mapeamento: Índice previsto {e} não encontrado em reverse_mapping_dict.")
         sys.exit(1)
    except Exception as e:
        print(f"Erro ao mapear previsões ou extrair probabilidades: {e}")
        sys.exit(1)

    # Montar DataFrame
    df_final_predictions = df_ids.copy()
    df_final_predictions['Situacao_Prevista'] = predicted_class_names
    df_final_predictions['Probabilidade_Prevista'] = [f"{p:.2%}" for p in predicted_probabilities]
    
    # Adicionar Top 3 features SHAP se disponíveis
    if include_shap:
        print("Identificando Top 3 features SHAP para cada previsão...")
        
        # Inicializar colunas para Top 3 features
        for i in range(1, 4):
            df_final_predictions[f'Top{i}_Pos'] = ""
            df_final_predictions[f'Top{i}_Neg'] = ""
        
        # Verificar o formato dos valores SHAP
        shap_shape = np.array(shap_values).shape
        print(f"Formato dos valores SHAP: {shap_shape}")
        
        # Tratar diferentes formatos de valores SHAP
        if isinstance(shap_values, list):
            print("Valores SHAP estão em formato de lista (um array por classe).")
            
            # Para cada amostra, identificar as Top 3 features
            for i, (idx, row) in enumerate(X_new.iterrows()):
                # Obter o índice da classe prevista
                class_idx = int(y_pred_indices[i])
                
                # Verificar se o índice da classe está dentro do range de shap_values
                if class_idx >= len(shap_values):
                    print(f"Aviso: Índice de classe {class_idx} está fora do range de shap_values (0-{len(shap_values)-1})")
                    # Usar a primeira classe como fallback
                    class_idx = 0
                
                # Obter os valores SHAP para a classe prevista e a amostra atual
                shap_for_sample = shap_values[class_idx][i]
                top_pos, top_neg = get_top_features_from_shap_values(
                    shap_for_sample, 
                    list(X_new.columns), 
                    top_n=3
                )
                
                # Formatar e adicionar ao DataFrame
                for j, (feature, value) in enumerate(top_pos):
                    if j < 3:  # Garantir que não ultrapasse o Top 3
                        formatted_feature = format_shap_feature(feature, value, row)
                        df_final_predictions.at[idx, f'Top{j+1}_Pos'] = formatted_feature
                
                for j, (feature, value) in enumerate(top_neg):
                    if j < 3:  # Garantir que não ultrapasse o Top 3
                        formatted_feature = format_shap_feature(feature, value, row)
                        df_final_predictions.at[idx, f'Top{j+1}_Neg'] = formatted_feature
        
        elif len(shap_shape) == 3:
            print("Valores SHAP estão em formato de array 3D (amostras, features, classes).")
            # Formato (amostras, features, classes)
            
            # Para cada amostra, identificar as Top 3 features
            for i, (idx, row) in enumerate(X_new.iterrows()):
                # Obter o índice da classe prevista
                class_idx = int(y_pred_indices[i])
                
                # Verificar se o índice da classe está dentro do range
                if class_idx >= shap_shape[2]:
                    print(f"Aviso: Índice de classe {class_idx} está fora do range (0-{shap_shape[2]-1})")
                    # Usar a primeira classe como fallback
                    class_idx = 0
                
                # Obter os valores SHAP para a amostra atual e a classe prevista
                # Formato: shap_values[amostra, feature, classe]
                shap_for_sample = shap_values[i, :, class_idx]
                
                top_pos, top_neg = get_top_features_from_shap_values(
                    shap_for_sample, 
                    list(X_new.columns), 
                    top_n=3
                )
                
                # Formatar e adicionar ao DataFrame
                for j, (feature, value) in enumerate(top_pos):
                    if j < 3:  # Garantir que não ultrapasse o Top 3
                        formatted_feature = format_shap_feature(feature, value, row)
                        df_final_predictions.at[idx, f'Top{j+1}_Pos'] = formatted_feature
                
                for j, (feature, value) in enumerate(top_neg):
                    if j < 3:  # Garantir que não ultrapasse o Top 3
                        formatted_feature = format_shap_feature(feature, value, row)
                        df_final_predictions.at[idx, f'Top{j+1}_Neg'] = formatted_feature
        
        else:
            print(f"Formato de valores SHAP não reconhecido: {shap_shape}. Pulando análise SHAP.")
            include_shap = False
        
        if include_shap:
            print("Top 3 features SHAP adicionadas ao relatório.")
    else:
        print("Pulando adição de features SHAP ao relatório.")
    
    print("DataFrame final montado.")

    # == ETAPA 6: Salvar Relatório Final ==
    print("\n--- Etapa 6: Salvando Relatório Final ---")
    output_csv_path = os.path.join(paths["project_root"], "previsoes_novos_alunos_unificado.csv")
    try:
        df_final_predictions.to_csv(output_csv_path, index=False, sep=";", encoding="latin1")
        print(f"Relatório de previsões unificado salvo com sucesso em: {output_csv_path}")
    except Exception as e:
        print(f"Erro ao salvar o relatório CSV unificado: {e}")

    print("\nProcesso UNIFICADO de previsão concluído.")
