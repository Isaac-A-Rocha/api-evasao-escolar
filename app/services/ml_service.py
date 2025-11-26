import joblib
import os
import sys
from app.core.config import settings

class MLService:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """Carrega o modelo do disco na inicialização."""
        if not os.path.exists(settings.MODEL_PATH):
            raise FileNotFoundError(f"Modelo não encontrado em: {settings.MODEL_PATH}. Treine o modelo primeiro!")
        
        try:
            print(f"Carregando modelo de: {settings.MODEL_PATH}")
            self.model = joblib.load(settings.MODEL_PATH)
        except Exception as e:
            print(f"Erro ao carregar o modelo: {e}")
            raise e

    def predict_proba(self, features):
        """
        Recebe um DataFrame ou Array 2D e retorna a probabilidade da classe positiva (1).
        """
        if not self.model:
            raise RuntimeError("O modelo não está carregado.")
        
        # predict_proba retorna [[prob_0, prob_1], ...]. Queremos a coluna 1.
        return self.model.predict_proba(features)[:, 1]

# Instância global do serviço (Singleton)
ml_service = MLService()