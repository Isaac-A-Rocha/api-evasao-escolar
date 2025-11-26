from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# Schema de Entrada (Input)
class StudentInput(BaseModel):
    idade: int = Field(..., description="Idade do aluno", example=19)
    sexo: Literal['M', 'F'] = Field(..., description="Sexo do aluno (M ou F)", example="F")
    tipo_escola_medio: Literal['publica', 'privada'] = Field(..., description="Tipo de escola do ensino médio", example="publica")
    nota_enem: float = Field(..., description="Nota média do ENEM", ge=0, le=1000, example=650.5)
    renda_familiar: float = Field(..., description="Renda familiar mensal", example=2500.0)
    trabalha: int = Field(..., description="0 se não trabalha, 1 se trabalha", ge=0, le=1, example=1)
    horas_trabalho_semana: int = Field(..., description="Horas trabalhadas por semana", example=30)
    reprovacoes_1_sem: int = Field(..., description="Número de reprovações no 1º semestre", example=2)
    bolsista: int = Field(..., description="0 se não é bolsista, 1 se é", ge=0, le=1, example=0)
    distancia_campus_km: float = Field(..., description="Distância da residência até o campus em KM", example=12.3)

# Schema para Batch (Lista de alunos)
class StudentBatchInput(BaseModel):
    alunos: List[StudentInput]

# Schema de Saída (Output)
class PredictionOutput(BaseModel):
    prob_evasao: float = Field(..., description="Probabilidade de evasão (0.0 a 1.0)")
    classe_prevista: int = Field(..., description="1 = Evasão, 0 = Permanência")
    threshold: float = Field(..., description="Limiar de decisão utilizado")