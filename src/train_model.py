import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    roc_curve, 
    confusion_matrix,
    ConfusionMatrixDisplay
)

# --- 1. CONFIGURAÇÃO E CARREGAMENTO ---
DATA_PATH = "data/alunos.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_model.pkl")

# Garantir que a pasta do modelo existe
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Carregando dados de {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

# --- 2. PRÉ-PROCESSAMENTO (Limpeza e Codificação) ---

# Separar Features (X) e Alvo (y)
X = df.drop(columns=["evasao_ate_1ano"])
y = df["evasao_ate_1ano"]

# Codificar variáveis categóricas (One-Hot Encoding / Get Dummies)
# drop_first=True é importante para Regressão Logística para evitar multicolinearidade
# Variáveis afetadas: sexo (M/F), tipo_escola_medio (publica/privada)
X = pd.get_dummies(X, columns=["sexo", "tipo_escola_medio"], drop_first=True)

# Nota: Como os dados são sintéticos e limpos, não há nulos para tratar (fillna).
# Se houvesse, faríamos: X = X.fillna(0) ou pela média.

# Divisão Treino / Teste (70/30, estratificado)
print("Dividindo dados em treino e teste...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# --- 3. TREINAMENTO DO MODELO ---
print("Treinando modelo de Regressão Logística...")
# max_iter aumentado para garantir convergência
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# --- 4. AVALIAÇÃO ---
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1] # Probabilidade da classe positiva (1)

# Métricas
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n--- RESULTADOS DA AVALIAÇÃO ---")
print(f"Acurácia:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}")

# Plotar Curva ROC (Requisito obrigatório)
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Regressão Logística (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--') # Linha base (chute aleatório)
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR - Recall)')
plt.title('Curva ROC - Evasão Escolar')
plt.legend(loc='lower right')
plt.grid()

# Salvar gráfico (opcional, mas bom para relatórios) ou mostrar
plt.savefig(os.path.join(MODEL_DIR, "roc_curve.png"))
print(f"Gráfico ROC salvo em: {os.path.join(MODEL_DIR, 'roc_curve.png')}")
# plt.show() # Descomente se quiser que a janela abra na hora

# --- 5. SALVAR O MODELO ---
print("\nSalvando modelo...")
# Importante: Em produção real, deveríamos salvar também o pré-processador (encoders).
# Aqui salvaremos apenas o modelo treinado conforme pedido básico.
joblib.dump(model, MODEL_PATH)
print(f"Modelo salvo com sucesso em: {MODEL_PATH}")