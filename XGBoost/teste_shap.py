import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# === 1. Carregamento do modelo e dados
modelo = joblib.load("modelo_xgboost_final.pkl")
df = pd.read_csv("planilha_final.csv", sep=';', encoding='latin1')

# === 2. Pré-processamento igual ao modelo
df = df[~df['Situação (código)'].isin(df['Situação (código)'].value_counts()[df['Situação (código)'].value_counts()<20].index)]
df['Situação (código)'] = df['Situação (código)'].replace({3: 2, 12: 2})

drop_cols = ['Matrícula', 'Nome', 'Curso', 'Renda', 'Sexo', 'Bairro', 'Cidade',
             'Turma Atual', 'Pend. Financ.', 'Situação', 'Descrição']
X = df.drop(columns=drop_cols + ['Situação (código)'])

# === 3. Aplicação do modelo
probs = modelo.predict_proba(X)
evaded_class = modelo.classes_.tolist().index(2)  # índice da classe 'CAN'

# === 4. Identifica os 3 alunos com maior probabilidade de evasão (classe 2)
top_idxs = probs[:, evaded_class].argsort()[::-1][:3]

# === 5. Explicador SHAP
explainer = shap.Explainer(modelo, X)
shap_values = explainer(X)

# === 6. Gráfico de importância global
shap.plots.bar(shap_values, show=False)
plt.tight_layout()
plt.savefig("shap_importancia_global.png")
plt.close()

# === 7. Gera gráfico de explicação individual (waterfall) para os 3 com maior risco
for i, idx in enumerate(top_idxs):
    shap.plots.waterfall(shap_values[idx], show=False)
    plt.tight_layout()
    plt.savefig(f"shap_aluno_risco_{i+1}.png")
    plt.close()
