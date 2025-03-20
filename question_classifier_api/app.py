from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, List, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import io

app = FastAPI(
    title="API de Classificação de Perguntas",
    description="API para treinamento e classificação de perguntas em categorias",
    version="1.0.0"
)

# Definindo caminhos para os arquivos de modelo
MODEL_PATH = 'model/'
MODEL_FILE = os.path.join(MODEL_PATH, 'classifier.pkl')

# Garantir que o diretório para salvar o modelo existe
os.makedirs(MODEL_PATH, exist_ok=True)

# Modelo de dados para requisição de classificação
class QuestionRequest(BaseModel):
    pergunta: str

# Modelo de dados para resposta de classificação
class ClassificationResponse(BaseModel):
    pergunta: str
    categoria: str
    probabilidades: Optional[Dict[str, float]] = None

# Modelo de dados para resposta de treinamento
class TrainingResponse(BaseModel):
    message: str
    accuracy: float
    classification_report: Dict[str, Any]
    best_params: Dict[str, Any]

@app.post("/train", response_model=TrainingResponse)
async def train_model(file: UploadFile = File(...)):
    """
    Treina um modelo de classificação usando um arquivo CSV enviado.
    
    O arquivo deve conter pelo menos as colunas 'pergunta' e 'tag'.
    
    - **file**: Arquivo CSV com dados de treinamento
    
    Retorna métricas de desempenho e os melhores parâmetros do modelo.
    """
    try:
        # Verificar se o arquivo tem extensão válida
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="O arquivo deve estar no formato CSV")
        
        # Ler o arquivo CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Verificar se as colunas necessárias estão presentes
        required_columns = ['pergunta', 'tag']
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(status_code=400, detail=f"O arquivo deve conter as colunas: {required_columns}")
        
        # Separar os dados em features e target
        X = df['pergunta']
        y = df['tag']
        
        # Dividir os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Criar o pipeline de processamento e classificação
        # Vamos testar diferentes modelos e parâmetros para encontrar o melhor
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', SVC())
        ])
        
        # Definir os parâmetros para busca em grade
        param_grid = [
            {
                'tfidf__max_features': [None, 5000, 10000],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'classifier': [MultinomialNB()],
                'classifier__alpha': [0.01, 0.1, 1.0]
            },
            {
                'tfidf__max_features': [None, 5000, 10000],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'classifier': [SVC(probability=True)],
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__kernel': ['linear']
            },
            {
                'tfidf__max_features': [None, 5000],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'classifier': [RandomForestClassifier()],
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 10, 20]
            },
            {
                'tfidf__max_features': [None, 5000],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'classifier': [LogisticRegression()],
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__max_iter': [1000]
            }
        ]
        
        # Realizar a busca em grade
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        # Avaliar o modelo com os dados de teste
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        # Calcular métricas de desempenho
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Salvar o modelo treinado
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(best_model, f)
        
        return {
            'message': 'Modelo treinado e salvo com sucesso',
            'accuracy': accuracy,
            'classification_report': report,
            'best_params': {k: str(v) for k, v in grid_search.best_params_.items()}
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify", response_model=ClassificationResponse)
async def classify_question(question_request: QuestionRequest):
    """
    Classifica uma pergunta usando o modelo treinado.
    
    - **pergunta**: Texto da pergunta a ser classificada
    
    Retorna a categoria prevista e as probabilidades de cada categoria.
    """
    try:
        # Verificar se o modelo existe
        if not os.path.exists(MODEL_FILE):
            raise HTTPException(status_code=404, detail="Modelo não encontrado. Treine o modelo primeiro.")
        
        question = question_request.pergunta
        
        # Carregar o modelo treinado
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        
        # Classificar a pergunta
        category = model.predict([question])[0]
        
        # Obter as probabilidades de classificação (se o modelo suportar)
        try:
            probabilities = model.predict_proba([question])[0]
            classes = model.classes_
            proba_dict = {class_name: float(prob) for class_name, prob in zip(classes, probabilities)}
        except:
            proba_dict = {}
        
        return {
            'pergunta': question,
            'categoria': category,
            'probabilidades': proba_dict
        }
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Verifica se a API está funcionando.
    
    Retorna o status da API.
    """
    return {"status": "OK"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)