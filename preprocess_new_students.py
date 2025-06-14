#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para pré-processar os dados de novos alunos para previsão.
(Versão v2 - Não depende do arquivo do modelo para ordem das features)
Lê o arquivo 'alunos_para_prever.csv', seleciona e ordena as features
necessárias para o modelo XGBoost.
"""
import pandas as pd
import os
import sys

# --- Configuração de Caminhos (Assumindo execução da raiz do projeto) ---
def get_project_paths_from_root():
    """Determina os caminhos importantes assumindo execução da raiz."""
    paths = {}
    try:
        paths["project_root"] = os.path.abspath(os.getcwd()) # Assume execução da raiz
        print(f"Diretório raiz detectado: {paths["project_root"]}")
        return paths
    except Exception as e:
        print(f"Erro ao determinar caminhos a partir da raiz: {e}")
        return None

paths = get_project_paths_from_root()
if paths is None:
    sys.exit(1)

# --- Ordem Esperada das Features (Baseada nos scripts anteriores) ---
# Esta ordem deve corresponder EXATAMENTE à usada no treinamento do modelo XGBoost
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

# --- Pré-processamento --- 
if __name__ == "__main__":
    print("Iniciando pré-processamento de novos alunos (v2)...")
    
    # 1. Definir ordem esperada das features
    feature_order = EXPECTED_FEATURE_ORDER
    print(f"Ordem esperada das features: {feature_order}")

    # 2. Ler arquivo de novos alunos
    input_csv_path = os.path.join(paths["project_root"], "alunos_para_prever.csv")
    if not os.path.exists(input_csv_path):
        print(f"Erro: Arquivo 'alunos_para_prever.csv' não encontrado em {paths['project_root']}")
        sys.exit(1)
        
    try:
        # Tentar ler com latin1 primeiro
        df_new = pd.read_csv(input_csv_path, encoding="latin1", sep=";")
        print(f"Dados de novos alunos carregados (latin1): {df_new.shape}")
    except UnicodeDecodeError:
        try:
            # Tentar com utf-8 como fallback
            df_new = pd.read_csv(input_csv_path, encoding="utf-8", sep=";")
            print(f"Dados de novos alunos carregados (utf-8): {df_new.shape}")
        except Exception as e:
            print(f"Erro ao carregar {input_csv_path} com latin1 e utf-8: {e}")
            sys.exit(1)
    except Exception as e:
         print(f"Erro ao carregar {input_csv_path}: {e}")
         sys.exit(1)
        
    # Manter cópia das colunas de identificação (ajuste se precisar de outras)
    id_cols_to_keep = ['Matrícula', 'Nome', 'Curso']
    # Verificar se as colunas de ID existem antes de tentar acessá-las
    actual_id_cols = [col for col in id_cols_to_keep if col in df_new.columns]
    if not actual_id_cols:
        print("Aviso: Nenhuma coluna de identificação (Matrícula, Nome, Curso) encontrada.")
        df_ids = pd.DataFrame(index=df_new.index) # Cria DataFrame vazio com mesmo índice
    else:
         df_ids = df_new[actual_id_cols].copy()

    # 3. Verificar se todas as features necessárias existem
    missing_cols = [col for col in feature_order if col not in df_new.columns]
    if missing_cols:
        print(f"Erro: Colunas necessárias ausentes no arquivo 'alunos_para_prever.csv': {missing_cols}")
        print(f"Colunas encontradas: {list(df_new.columns)}")
        sys.exit(1)
        
    # 4. Selecionar e reordenar as colunas
    try:
        X_new = df_new[feature_order].copy()
        print(f"Features selecionadas e reordenadas: {X_new.shape}")
    except Exception as e:
        print(f"Erro ao selecionar/reordenar features: {e}")
        sys.exit(1)
        
    # 5. Verificar tipos de dados e tratar NaNs se necessário
    if X_new.isnull().values.any():
        print("Aviso: Existem valores nulos (NaN) nas features selecionadas.")
        # Aplicar uma estratégia de preenchimento. Ex: preencher com 0
        # ATENÇÃO: Escolha a estratégia de preenchimento com cuidado!
        # Usar 0 pode não ser ideal para todas as colunas.
        # Considere usar mediana, média ou um valor específico se fizer mais sentido.
        print("Aplicando preenchimento de NaNs com 0 (Atenção: revise se esta é a melhor estratégia)...")
        X_new.fillna(0, inplace=True)
        if X_new.isnull().values.any():
             print("Erro: Falha ao preencher NaNs.")
             sys.exit(1)
        else:
             print("NaNs preenchidos com 0.")

    # 6. Salvar dados pré-processados
    output_dir = paths["project_root"] # Salvar na raiz
    output_path = os.path.join(output_dir, "X_novos_alunos_preprocessado.csv")
    ids_output_path = os.path.join(output_dir, "ids_novos_alunos.csv")
    try:
        X_new.to_csv(output_path, index=False, sep=";", encoding="latin1")
        df_ids.to_csv(ids_output_path, index=False, sep=";", encoding="latin1")
        print(f"Dados pré-processados salvos em: {output_path}")
        print(f"IDs correspondentes salvos em: {ids_output_path}")
    except Exception as e:
        print(f"Erro ao salvar dados pré-processados: {e}")

    print("Pré-processamento de novos alunos concluído.")
