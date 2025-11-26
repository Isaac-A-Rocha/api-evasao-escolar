API de Previsão de Evasão Escolar (Machine Learning + FastAPI)Este projeto consiste em uma solução completa de Data Science e Engenharia de Software para prever a probabilidade de evasão escolar de alunos. O sistema utiliza um modelo de Regressão Logística treinado com dados sintéticos e disponibiliza essas previsões através de uma API RESTful construída com FastAPI seguindo o padrão de arquitetura MVC (Model-View-Controller). 

Funcionalidades Geração de Dados: Script para criar datasets sintéticos de alunos.Treinamento de Modelo: Pipeline de limpeza, transformação (One-Hot Encoding) e treinamento de modelo.API REST:Health Check: Verifica status da API.Previsão Individual: Recebe dados de um aluno e retorna a probabilidade de evasão.Previsão em Lote (Batch): Processa múltiplos alunos em uma única requisição.Documentação Automática: Swagger UI e ReDoc integrados. Estrutura do Projeto (MVC)O projeto foi organizado para garantir a separação de responsabilidades:TRABALHO 3/
├── app/
│   ├── main.py              # Ponto de entrada da aplicação (Entrypoint)
│   ├── core/                # Configurações globais (caminhos, variáveis de ambiente)
│   ├── controllers/         # Regras de negócio e orquestração dos dados
│   ├── schemas/             # Contratos de dados (Pydantic) para validação de I/O
│   ├── services/            # Serviços de infraestrutura (Carregamento do modelo ML)
│   └── views/               # Definição das Rotas (Endpoints)
├── data/                    # Armazenamento de datasets (alunos.csv)
├── model/                   # Artefatos do modelo (arquivo .pkl e gráficos)
├── src/                     # Scripts de automação (geração de dados e treinamento)
└── requirements.txt         # Lista de dependências do projeto
 Instalação e ConfiguraçãoPré-requisitosPython 3.8 ou superior.1. Clonar e preparar o ambienteNo terminal, na raiz do projeto:Bash# 1. Criar ambiente virtual
python -m venv venv

# 2. Ativar ambiente virtual
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Instalar dependências
pip install -r requirements.txt
(Certifique-se de que o arquivo requirements.txt contém: pandas, numpy, scikit-learn, joblib, fastapi, uvicorn, pydantic) Etapa 1: Machine LearningAntes de iniciar a API, é necessário gerar os dados e treinar o modelo.Gerar o Dataset:Bashpython src/generate_csv.py
Isso criará o arquivo data/alunos.csv.Treinar o Modelo:Bashpython src/train_model.py
Isso salvará o modelo treinado em model/logistic_model.pkl e gerará a curva ROC. Etapa 2: Executar a APICom o modelo treinado, inicie o servidor de desenvolvimento:Bashuvicorn app.main:app --reload
O servidor iniciará em: http://127.0.0.1:8000 Documentação da APIAcesse a documentação interativa para testar os endpoints diretamente pelo navegador:Swagger UI: http://127.0.0.1:8000/docsReDoc: http://127.0.0.1:8000/redocEndpoints PrincipaisMétodoRotaDescriçãoGET/healthVerifica se a API está online e o modelo carregado.POST/predictPrevisão de risco para um único aluno.POST/predict_batchPrevisão de risco para uma lista de alunos.Exemplo de Requisição (JSON)Para testar no endpoint /predict:JSON{
  "idade": 19,
  "sexo": "F",
  "tipo_escola_medio": "publica",
  "nota_enem": 650.5,
  "renda_familiar": 2500.0,
  "trabalha": 1,
  "horas_trabalho_semana": 30,
  "reprovacoes_1_sem": 0,
  "bolsista": 1,
  "distancia_campus_km": 15.0
}
 Métricas do ModeloO modelo de Regressão Logística é avaliado com as seguintes métricas (exibidas no terminal após o treino):AcuráciaPrecisionRecallF1-ScoreAUC-ROCA Curva ROC gerada pode ser visualizada em model/roc_curve.png.👨‍💻 AutorDesenvolvido como parte do Trabalho 3 da disciplina de RP.
