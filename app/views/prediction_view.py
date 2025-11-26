from fastapi import APIRouter
from app.schemas.student_schema import StudentInput, StudentBatchInput, PredictionOutput
from app.controllers.prediction_controller import prediction_controller
from app.services.ml_service import ml_service

router = APIRouter()

@router.get("/health")
def health_check():
    """Verifica se a API está online e o modelo carregado."""
    model_status = ml_service.model is not None
    return {
        "status": "ok", 
        "model_loaded": model_status,
        "api_version": "1.0.0"
    }

@router.post("/predict", response_model=PredictionOutput)
def predict_student(student: StudentInput):
    """
    Endpoint para prever a probabilidade de evasão de UM aluno.
    """
    return prediction_controller.predict_single(student)

@router.post("/predict_batch", response_model=list[PredictionOutput])
def predict_batch_students(batch: StudentBatchInput):
    """
    Endpoint para prever vários alunos de uma vez.
    """
    return prediction_controller.predict_batch(batch.alunos)