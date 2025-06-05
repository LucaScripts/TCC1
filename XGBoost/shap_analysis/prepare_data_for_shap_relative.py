#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para preparar os dados para análise SHAP do modelo XGBoost 
(versão v3 com correção no nome da variável importada e ajuste no sys.path)
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sys
import os

# Tenta encontrar o diretório raiz do projeto (TCC1) de forma mais robusta
try:
    # Caminho do script atual (shap_analysis)
    script_path = os.path.abspath(__file__)
    shap_analysis_dir = os.path.dirname(script_path)
    # Subir dois níveis para chegar à raiz do projeto (TCC1)
    project_root = os.path.abspath(os.path.join(shap_analysis_dir, "..", ".."))
    
    # Verificar se o mapeamento_classes.py existe na raiz encontrada
    mapping_file_path = os.path.join(project_root, "mapeamento_classes.py")
    if not os.path.exists(mapping_file_path):
        raise ImportError(f"Arquivo 'mapeamento_classes.py' não encontrado em {project_root}")
        
    # Adicionar a raiz ao sys.path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Adicionado ao sys.path: {project_root}")
        
    # Tentar importar novamente com o nome CORRETO da variável
    # CORREÇÃO: Importar 'mapping_dict' em vez de 'mapeamento_classes'
    from mapeamento_classes import mapping_dict, reverse_mapping_dict
    print("Importação de 'mapping_dict' e 'reverse_mapping_dict' bem-sucedida.")

except ImportError as e:
    print(f"Erro Crítico: Falha ao configurar o sys.path ou importar de 'mapeamento_classes.py'. {e}")
    print("Verifique se este script está em 'TCC1/XGBoost/shap_analysis/' e 'mapeamento_classes.py' está em 'TCC1/' e contém as variáveis 'mapping_dict' e 'reverse_mapping_dict'.")
    sys.exit(1)
except Exception as e:
    print(f"Erro inesperado ao configurar path: {e}")
    sys.exit(1)

def preparar_dados():
    print("Preparando dados para análise SHAP (v3 - caminhos relativos)...")
    
    # 1. Carregar os dados (esperado na pasta raiz TCC1)
    csv_path = os.path.join(project_root, "planilha_final.csv")
    if not os.path.exists(csv_path):
        print(f"Erro: Arquivo 'planilha_final.csv' não encontrado em {project_root}")
        return None, None, None
        
    try:
        df = pd.read_csv(csv_path, encoding="latin1", sep=";")
        print(f"Dados carregados: {df.shape[0]} linhas e {df.shape[1]} colunas")
    except Exception as e:
        print(f"Erro ao carregar {csv_path}: {e}")
        return None, None, None
    
    # 2. Remover classes com poucos exemplos
    classe_counts = df["Situação (código)"].value_counts()
    classes_para_remover = classe_counts[classe_counts < 10].index
    df = df[~df["Situação (código)"].isin(classes_para_remover)]
    print(f"Após remover classes raras: {df.shape[0]} linhas")
    
    # 3. Agrupar CAU (3) para CAN (2)
    agrupamento_classes = {3: 2}
    df["Situação (código)"] = df["Situação (código)"].replace(agrupamento_classes)
    
    # 4. Aplicar o mapeamento inicial (pode ter gaps)
    # CORREÇÃO: Usar 'mapping_dict' importado
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
    colunas_para_remover = ["Matrícula", "Nome", "Curso", "Renda", "Sexo",
                            "Bairro", "Cidade", "Turma Atual",
                            "Pend. Financ.", "Situação", "Descrição"]
    X = X.drop(columns=[col for col in colunas_para_remover if col in X.columns])
    
    # --- Mapeamento Final Contíguo (0 a N-1) --- 
    final_le = LabelEncoder()
    final_le.fit(y_inicial)
    y = final_le.transform(y_inicial)
    
    final_classes_indices = final_le.classes_
    final_class_names = [reverse_mapping_dict[idx] for idx in final_classes_indices]
    num_final_classes = len(final_classes_indices)
    mapeamento_final = dict(zip(range(num_final_classes), final_class_names))
    
    print(f"Labels iniciais presentes: {final_classes_indices}")
    print(f"Labels finais contíguos: {np.unique(y)}")
    print(f"Nomes das classes finais: {final_class_names}")
    print(f"Número de classes finais: {num_final_classes}")
    
    # Salvar os dados preparados na pasta atual (shap_analysis)
    output_dir = os.path.dirname(script_path) # Diretório do script atual
    X_output_path = os.path.join(output_dir, "X_prepared.csv")
    y_output_path = os.path.join(output_dir, "y_prepared.csv")
    mapping_output_path = os.path.join(output_dir, "class_mapping_info.txt")
    
    try:
        X.to_csv(X_output_path, index=False)
        pd.Series(y).to_csv(y_output_path, index=False)
        
        with open(mapping_output_path, "w") as f:
            f.write(f"Labels iniciais presentes: {final_classes_indices}\n")
            f.write(f"Labels finais contíguos: {list(range(num_final_classes))}\n")
            f.write(f"Nomes das classes finais: {final_class_names}\n")
            f.write(f"Número de classes finais: {num_final_classes}\n")
            f.write("\nMapeamento final (índice -> nome da classe):\n")
            for idx, name in mapeamento_final.items():
                f.write(f"{idx}: {name}\n")
        
        print(f"Dados preparados e salvos em {output_dir}")
        return X, y, mapeamento_final
        
    except Exception as e:
        print(f"Erro ao salvar arquivos preparados: {e}")
        return None, None, None

if __name__ == "__main__":
    preparar_dados()
