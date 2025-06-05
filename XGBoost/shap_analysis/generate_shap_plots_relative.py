#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para gerar gráficos SHAP (summary plots) a partir dos valores calculados (versão com caminhos relativos).
"""
import shap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg") # Use backend não interativo
import matplotlib.pyplot as plt
import os
import re
import sys

def gerar_graficos_shap():
    print("Iniciando geração dos gráficos SHAP (caminhos relativos)...")
    
    # 1. Definir diretório base (pasta atual do script)
    script_dir = os.path.dirname(__file__)
    data_dir = script_dir # Ler e salvar na mesma pasta
    
    X_path = os.path.join(data_dir, "X_prepared.csv")
    shap_values_path = os.path.join(data_dir, "shap_values.npy")
    mapping_info_path = os.path.join(data_dir, "class_mapping_info.txt")
    output_dir = data_dir # Salvar gráficos na pasta atual
    
    # Verificar se os arquivos necessários existem
    if not all(os.path.exists(p) for p in [X_path, shap_values_path, mapping_info_path]):
        print(f"Erro: Arquivos necessários (X_prepared.csv, shap_values.npy, class_mapping_info.txt) não encontrados em {data_dir}.")
        return
        
    # 2. Carregar dados e valores SHAP
    try:
        X = pd.read_csv(X_path)
        shap_values = np.load(shap_values_path)
        print(f"Dados X carregados: {X.shape}")
        print(f"Valores SHAP carregados: {shap_values.shape}")
    except Exception as e:
        print(f"Erro ao carregar dados ou valores SHAP: {e}")
        return
        
    # 3. Ler nomes das classes do arquivo de mapeamento
    try:
        with open(mapping_info_path, "r") as f:
            content = f.read()
            match = re.search(r"Nomes das classes finais: \[(.*?)\]", content)
            if match:
                class_names_str = match.group(1)
                class_names = [name.strip().strip("'") for name in class_names_str.split(",")]
                print(f"Nomes das classes lidos: {class_names}")
                # Verificação da dimensão SHAP (ndarray pode ter shape (samples, features, classes) ou (classes, samples, features))
                shap_dim_classes = -1
                if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                    shap_dim_classes = shap_values.shape[2]
                elif isinstance(shap_values, list):
                    shap_dim_classes = len(shap_values)
                
                if shap_dim_classes != len(class_names):
                     print(f"Erro: Número de nomes de classes ({len(class_names)}) não corresponde à dimensão dos valores SHAP ({shap_dim_classes}).")
                     return
            else:
                print("Erro: Não foi possível encontrar 'Nomes das classes finais' no arquivo de mapeamento.")
                return
    except Exception as e:
        print(f"Erro ao ler o arquivo de mapeamento de classes: {e}")
        return

    # 4. Gerar gráficos SHAP
    
    # Gráfico de barras geral (importância média absoluta)
    try:
        print("Gerando gráfico de barras SHAP geral (importância média absoluta)...")
        plt.figure()
        # Para ndarray (samples, features, classes), o summary_plot lida com isso
        shap.summary_plot(shap_values, X, plot_type="bar", class_names=class_names, show=False)
        plt.title("Importância Média Absoluta SHAP (Geral)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_summary_bar_geral.png"))
        plt.close()
        print("Gráfico de barras geral salvo.")
    except Exception as e:
        print(f"Erro ao gerar gráfico de barras geral: {e}")

    # Gráficos por classe (bar e beeswarm)
    for i, class_name in enumerate(class_names):
        safe_class_name = class_name.replace('/', '_') # Sanitize filename
        print(f"Gerando gráficos para a classe: {class_name} (Índice {i})")
        
        # Extrair SHAP values para a classe i
        # Se for ndarray (samples, features, classes)
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values_class = shap_values[:,:,i]
        # Se for lista de arrays (um por classe)
        elif isinstance(shap_values, list):
            shap_values_class = shap_values[i]
        else:
            print(f"  - Erro: Formato SHAP values não suportado para extração por classe.")
            continue
            
        # Gráfico de Barras por Classe
        try:
            plt.figure()
            shap.summary_plot(shap_values_class, X, plot_type="bar", show=False)
            plt.title(f"Importância SHAP (Bar) - Classe: {class_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"shap_summary_bar_classe_{safe_class_name}.png"))
            plt.close()
            print(f"  - Gráfico de barras salvo para {class_name}.")
        except Exception as e:
            print(f"  - Erro ao gerar gráfico de barras para {class_name}: {e}")
            
        # Gráfico Beeswarm por Classe
        try:
            plt.figure()
            shap.summary_plot(shap_values_class, X, plot_type="dot", show=False) # 'dot' é o beeswarm
            plt.title(f"Impacto das Features (Beeswarm) - Classe: {class_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"shap_summary_beeswarm_classe_{safe_class_name}.png"))
            plt.close()
            print(f"  - Gráfico beeswarm salvo para {class_name}.")
        except Exception as e:
            print(f"  - Erro ao gerar gráfico beeswarm para {class_name}: {e}")

    print("Geração dos gráficos SHAP concluída.")

if __name__ == "__main__":
    gerar_graficos_shap()
