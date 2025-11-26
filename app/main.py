from fastapi import FastAPI
from app.core.config import settings
from app.views import prediction_view

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API para predição de evasão escolar usando Regressão Logística."
)

# Incluir rotas
app.include_router(prediction_view.router)

# Rota raiz opcional
@app.get("/")
def root():
    return {"message": "API de Evasão Escolar Online. Acesse /docs para documentação."}

if __name__ == "__main__":
    import uvicorn
    # Permite rodar como script python: python app/main.py
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)