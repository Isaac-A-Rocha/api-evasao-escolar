import os

class Settings:
    PROJECT_NAME: str = "API de Previsão de Evasão"
    VERSION: str = "1.0.0"
    
    # Caminho base do projeto (subindo 2 níveis a partir de app/core)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Caminho do modelo treinado
    # Ajuste o nome do arquivo se o seu for diferente (ex: logistic_model.pkl)
    MODEL_PATH = os.path.join(BASE_DIR, "model", "logistic_model.pkl")

settings = Settings()