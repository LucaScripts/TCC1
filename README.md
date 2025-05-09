# TCC\_MVP1 - Classificação de Risco Acadêmico

Este projeto tem como objetivo construir modelos de Machine Learning para **prever o risco acadêmico** de alunos, utilizando dados históricos de matrícula, frequência, desempenho e outros fatores.

Modelos utilizados:

* 🔥 **Random Forest**
* 🔥 **Regressão Logística**
* 🔥 **XGBoost**

---

## 📂 Estrutura de Pastas

```
TCC_MVP1/
├── Random Forest/
│   ├── modelo.py
│   ├── prever_em_lote.py
│   ├── prever_novo.py
│   ├── modelo_random_forest.pkl
│   └── resultados_previsoes.csv
├── Regressão Logística/
│   ├── modelo_logistico.py
│   ├── modelo_logistico.pkl
├── XGBoost/
│   ├── modelo_xgboost.py
│   ├── modelo_xgboost.pkl
├── planilha_final.csv
├── alunos_para_prever.csv
├── requirements.txt
```

---

## 🛠️ Como Rodar o Projeto

1. Clone o repositório:

   ```bash
   git clone https://github.com/seuusuario/TCC_MVP1.git
   ```

2. Acesse a pasta do projeto:

   ```bash
   cd TCC_MVP1
   ```

3. Crie e ative um ambiente virtual (opcional, mas recomendado):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

4. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

5. Rode o modelo desejado:

   * Random Forest:

     ```bash
     cd "Random Forest"
     python modelo.py
     ```
   * Regressão Logística:

     ```bash
     cd "../Regressão Logística"
     python modelo_logistico.py
     ```
   * XGBoost:

     ```bash
     cd "../XGBoost"
     python modelo_xgboost.py
     ```

---

## 📊 Tecnologias Usadas

* Python
* Pandas
* Scikit-learn
* Imbalanced-learn (SMOTE)
* XGBoost
* Matplotlib
* Joblib

---

## 🎯 Objetivo

O principal objetivo deste projeto é:

* **Identificar alunos em situação de risco acadêmico** de maneira preditiva.
* **Classificar alunos** em diferentes situações (cancelamento, conclusão, trancamento, etc.) com base nos dados fornecidos.

Este projeto é parte integrante do Trabalho de Conclusão de Curso (TCC) do curso de **Sistemas de Informação** no IFBA - Vitória da Conquista.

---

## ✍️ Autor

* **Lucas Dias da Silva**
  [LinkedIn](https://www.linkedin.com/in/seu-perfil) | [GitHub](https://github.com/seuusuario)

---

# 🚀

## Quer que eu também te mande um `README.md` com algumas imagens (ex.: o gráfico das classes)? 🎨

(pode deixar ele ainda mais visual e chamativo!)
Quer? 🎯
