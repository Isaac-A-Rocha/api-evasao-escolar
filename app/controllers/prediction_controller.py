import pandas as pd
from fastapi import HTTPException
from app.services.ml_service import ml_service
from app.schemas.student_schema import StudentInput, PredictionOutput

class PredictionController:
    
    def prepare_features(self, student: StudentInput) -> pd.DataFrame:
        """
        Transforma o objeto de entrada (JSON) no formato exato que o modelo espera (DataFrame).
        Realiza o One-Hot Encoding manual para garantir a ordem das colunas.
        """
        # 1. Criar dicionário base
        data = {
            "idade": [student.idade],
            "nota_enem": [student.nota_enem],
            "renda_familiar": [student.renda_familiar],
            "trabalha": [student.trabalha],
            "horas_trabalho_semana": [student.horas_trabalho_semana],
            "reprovacoes_1_sem": [student.reprovacoes_1_sem],
            "bolsista": [student.bolsista],
            "distancia_campus_km": [student.distancia_campus_km],
            
            # --- TRATAMENTO DE VARIÁVEIS DUMMY ---
            # No treino, usamos drop_first=True. 
            # Para 'sexo' (F, M), geralmente o 'F' é o primeiro (alfabético) e cai, ficando 'sexo_M'.
            # Para 'tipo_escola' (privada, publica), 'privada' cai, ficando 'tipo_escola_medio_publica'.
            # É CRUCIAL que esses nomes de colunas sejam IDÊNTICOS aos do X_train.columns
            
            "sexo_M": [1 if student.sexo == "M" else 0],
            "tipo_escola_medio_publica": [1 if student.tipo_escola_medio == "publica" else 0]
        }
        
        # Retorna DataFrame
        return pd.DataFrame(data)

    def predict_single(self, student: StudentInput) -> PredictionOutput:
        try:
            # 1. Preparar dados
            df_features = self.prepare_features(student)
            
            # 2. Chamar serviço de ML
            proba = ml_service.predict_proba(df_features)[0]
            
            # 3. Aplicar regra de negócio (Threshold)
            threshold = 0.5
            classe = 1 if proba >= threshold else 0
            
            # 4. Retornar objeto de saída
            return PredictionOutput(
                prob_evasao=round(float(proba), 4),
                classe_prevista=classe,
                threshold=threshold
            )
        except Exception as e:
            # Logar erro real no console e retornar 500 para o usuário
            print(f"Erro no controller: {e}")
            raise HTTPException(status_code=500, detail="Erro interno ao processar predição.")

    def predict_batch(self, students_list: list[StudentInput]) -> list[PredictionOutput]:
        results = []
        # Para performance real, o ideal seria criar um DataFrame gigante e prever tudo de uma vez.
        # Mas para simplificar a didática aqui, vamos iterar (o modelo é leve).
        for student in students_list:
            results.append(self.predict_single(student))
        return results

prediction_controller = PredictionController()