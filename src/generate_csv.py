import pandas as pd
import numpy as np
import os

# Garantir pasta data/
os.makedirs("data", exist_ok=True)

np.random.seed(42)
N = 300  # quantidade de alunos

# Variáveis (colunas)
idade = np.random.randint(17, 45, N)
sexo = np.random.choice(["M", "F"], N, p=[0.45, 0.55])
tipo_escola_medio = np.random.choice(["publica", "privada"], N, p=[0.75, 0.25])
nota_enem = np.random.normal(600, 120, N).clip(300, 800)
renda_familiar = np.random.randint(800, 15000, N)
trabalha = np.random.choice([0, 1], N, p=[0.55, 0.45])

# CORREÇÃO AQUI: Envolvemos a lista em np.array() para permitir operações matemáticas
horas_trabalho_semana = np.array([
    np.random.randint(0, 50) if t == 1 else 0 for t in trabalha
])

reprovacoes_1_sem = np.random.poisson(0.4, N).clip(0, 5)
bolsista = np.random.choice([0, 1], N, p=[0.70, 0.30])
distancia_campus_km = np.abs(np.random.normal(12, 10, N)).clip(0, 70)

# Alvo (evasão)
# Agora horas_trabalho_semana é um array numpy, então a comparação > 30 funciona
prob_evasao = (
    (idade > 30) * 0.15 +
    (trabalha == 1) * 0.25 +
    (horas_trabalho_semana > 30) * 0.20 + 
    (renda_familiar < 2000) * 0.20 +
    (reprovacoes_1_sem >= 2) * 0.30 +
    (tipo_escola_medio == "publica") * 0.10 +
    (distancia_campus_km > 30) * 0.15
)

# Converter probabilidade em 0/1
evasao_ate_1ano = np.random.binomial(1, prob_evasao.clip(0, 0.95))

df = pd.DataFrame({
    "idade": idade,
    "sexo": sexo,
    "tipo_escola_medio": tipo_escola_medio,
    "nota_enem": nota_enem,
    "renda_familiar": renda_familiar,
    "trabalha": trabalha,
    "horas_trabalho_semana": horas_trabalho_semana,
    "reprovacoes_1_sem": reprovacoes_1_sem,
    "bolsista": bolsista,
    "distancia_campus_km": distancia_campus_km,
    "evasao_ate_1ano": evasao_ate_1ano
})

df.to_csv("data/alunos.csv", index=False)

print("CSV gerado com sucesso: data/alunos.csv  (300 linhas)")