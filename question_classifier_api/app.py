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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

CATEGORY_MODEL_FILE = os.path.join(MODEL_PATH, 'category_classifier.pkl')
SUPPORT_MODEL_FILE = os.path.join(MODEL_PATH, 'support_classifier.pkl')

# Categorias esperadas
CATEGORIES = ['assinatura', 'ecampus', 'pag', 'sei', 'revista', 'wifi', 'mautic', 'metabase', 'evoto', 'outros']

# Garantir que o diretório para salvar o modelo existe
os.makedirs(MODEL_PATH, exist_ok=True)

# Modelo de dados para requisição de classificação
class QuestionRequest(BaseModel):
    pergunta: str

# Modelo de dados para resposta de classificação
class ClassificationResponse(BaseModel):
    pergunta: str
    categoria: str
    suporte_tecnico: bool
    probabilidades_categoria: Optional[Dict[str, float]] = None
    probabilidade_suporte: Optional[float] = None

# Modelo de dados para resposta de treinamento
class TrainingResponse(BaseModel):
    message: str
    accuracy_categoria: float
    accuracy_suporte: float
    classification_report_categoria: Dict[str, Any]
    classification_report_suporte: Dict[str, Any]
    best_params_categoria: Dict[str, Any]
    best_params_suporte: Dict[str, Any]

@app.post("/train", response_model=TrainingResponse)
async def train_model(file: UploadFile = File(...)):
    """
    Treina modelos de classificação para categoria e suporte técnico usando um arquivo CSV enviado.
    
    O arquivo deve conter pelo menos as colunas 'pergunta', 'tag' e 'suporte'.
    
    - **file**: Arquivo CSV com dados de treinamento
    
    Retorna métricas de desempenho e os melhores parâmetros dos modelos.
    """
    try:
        # Verificar se o arquivo tem extensão válida
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="O arquivo deve estar no formato CSV")
        
        # Ler o arquivo CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Verificar se as colunas necessárias estão presentes
        required_columns = ['pergunta', 'tag', 'suporte']
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(status_code=400, detail=f"O arquivo deve conter as colunas: {required_columns}")
        
        # Verificar se a coluna 'suporte' contém apenas valores binários (0 ou 1)
        if not set(df['suporte'].unique()).issubset({0, 1}):
            raise HTTPException(status_code=400, detail="A coluna 'suporte' deve conter apenas valores 0 ou 1")
        
        # Verificar se as colunas necessárias estão presentes
        required_columns = ['pergunta', 'tag']
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(status_code=400, detail=f"O arquivo deve conter as colunas: {required_columns}")
        
        # Filtra apenas as categorias esperadas
        df = df[df['tag'].isin(CATEGORIES)]

        # Separar os dados em features e target
        X = df['pergunta']
        y_categoria = df['tag']
        y_suporte = df['suporte']
        
        # ====================== TREINAMENTO DO MODELO DE CATEGORIAS ====================== #

        # Dividir os dados em treino e teste para categorias
        X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X, y_categoria, test_size=0.2, random_state=42)
        
        # Criar o pipeline para categoria
        pipeline_cat = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', SVC())
        ])
        
        # Parâmetros para o modelo de categoria
        param_grid_cat = [
            {
                'tfidf__max_features': [None, 5000, 10000],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'classifier': [MultinomialNB()],
                'classifier__alpha': [0.01, 0.1, 1.0]
            },
            {
                'tfidf__max_features': [None, 5000, 10000],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'classifier': [SVC()],
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__kernel': ['linear'],
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
                'classifier__solver': ['liblinear', 'lbfgs'],
                'classifier__penalty': ['l2'],
                'classifier__max_iter': [1000]
            },
            {
                'tfidf__max_features': [None, 5000],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'classifier': [GradientBoostingClassifier()],
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 7]
            }
        ]
        
        # Realizar a busca em grade para o modelo de categoria
        grid_search_cat = GridSearchCV(pipeline_cat, param_grid_cat, cv=5, n_jobs=-1, verbose=1)
        grid_search_cat.fit(X_train_cat, y_train_cat)

        # Avaliar o modelo de categoria
        best_model_cat = grid_search_cat.best_estimator_
        
        # Se o melhor modelo é um SVC, aplique calibração
        if isinstance(best_model_cat.named_steps['classifier'], SVC):
            calibrated_model_cat = Pipeline([
                ('tfidf', best_model_cat.named_steps['tfidf']),
                ('classifier', CalibratedClassifierCV(
                    best_model_cat.named_steps['classifier'],
                    method='sigmoid',
                    cv=5
                ))
            ])
            calibrated_model_cat.fit(X_train_cat, y_train_cat)
            best_model_cat = calibrated_model_cat
        
        y_pred_cat = best_model_cat.predict(X_test_cat)
        accuracy_cat = accuracy_score(y_test_cat, y_pred_cat)
        report_cat = classification_report(y_test_cat, y_pred_cat, output_dict=True)

        print(f"Melhor modelo para categoria: {best_model_cat}")
        print(f"Melhores parâmetros: {grid_search_cat.best_params_}")

        # Salvar o modelo de categoria
        with open(CATEGORY_MODEL_FILE, 'wb') as f:
            pickle.dump(best_model_cat, f)



        # ====================== TREINAMENTO DO MODELO DE SUPORTE TÉCNICO ====================== #
        
        # Dividir os dados em treino e teste para suporte
        X_train_sup, X_test_sup, y_train_sup, y_test_sup = train_test_split(X, y_suporte, test_size=0.2, random_state=42)
        
        # Criar o pipeline para suporte técnico
        pipeline_sup = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', LogisticRegression())
        ])
        
        # Parâmetros para o modelo de suporte técnico
        param_grid_sup = [
            {
                'tfidf__max_features': [None, 5000, 10000],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'classifier': [LogisticRegression(max_iter=1000)],
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__class_weight': [None, 'balanced']
            },
            {
                'tfidf__max_features': [None, 5000, 10000],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'classifier': [RandomForestClassifier()],
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__class_weight': [None, 'balanced']
            }
        ]
        
        # Realizar a busca em grade para o modelo de suporte
        grid_search_sup = GridSearchCV(pipeline_sup, param_grid_sup, cv=5, n_jobs=-1, verbose=1)
        grid_search_sup.fit(X_train_sup, y_train_sup)
        
        # Avaliar o modelo de suporte
        best_model_sup = grid_search_sup.best_estimator_
        y_pred_sup = best_model_sup.predict(X_test_sup)
        accuracy_sup = accuracy_score(y_test_sup, y_pred_sup)
        report_sup = classification_report(y_test_sup, y_pred_sup, output_dict=True)
        
        # Salvar o modelo de suporte
        with open(SUPPORT_MODEL_FILE, 'wb') as f:
            pickle.dump(best_model_sup, f)

        
        return {
            'message': 'Modelos treinados e salvos com sucesso',
            'accuracy_categoria': accuracy_cat,
            'accuracy_suporte': accuracy_sup,
            'classification_report_categoria': report_cat,
            'classification_report_suporte': report_sup,
            'best_params_categoria': {k: str(v) for k, v in grid_search_cat.best_params_.items()},
            'best_params_suporte': {k: str(v) for k, v in grid_search_sup.best_params_.items()}
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify", response_model=ClassificationResponse)
async def classify_question(question_request: QuestionRequest):
    """
    Classifica uma pergunta usando os modelos treinados.
    
    - **pergunta**: Texto da pergunta a ser classificada
    
    Retorna a categoria prevista, o status de suporte técnico e as probabilidades correspondentes.
    """
    try:
        # Verificar se os modelos existem
        if not os.path.exists(CATEGORY_MODEL_FILE) or not os.path.exists(SUPPORT_MODEL_FILE):
            raise HTTPException(status_code=404, detail="Modelos não encontrados. Treine os modelos primeiro.")
        
        question = question_request.pergunta
        
        # Carregar o modelo de categoria
        with open(CATEGORY_MODEL_FILE, 'rb') as f:
            category_model = pickle.load(f)
        
        # Carregar o modelo de suporte
        with open(SUPPORT_MODEL_FILE, 'rb') as f:
            support_model = pickle.load(f)
        
        # Classificar a pergunta - categoria
        category = category_model.predict([question])[0]
        
        # Classificar a pergunta - suporte técnico
        support = support_model.predict([question])[0]
        support_bool = bool(support)
        
        # Obter as probabilidades de classificação para categoria
        try:
            cat_probabilities = category_model.predict_proba([question])[0]
            cat_classes = category_model.classes_
            cat_proba_dict = {class_name: float(prob) for class_name, prob in zip(cat_classes, cat_probabilities)}
        except:
            cat_proba_dict = {}
        
        # Obter probabilidade de suporte técnico
        try:
            # Para modelos binários, geralmente a segunda classe (índice 1) representa a classe positiva (1)
            support_proba = float(support_model.predict_proba([question])[0][1])
        except:
            support_proba = None
        
        return {
            'pergunta': question,
            'categoria': category,
            'suporte_tecnico': support_bool,
            'probabilidades_categoria': cat_proba_dict,
            'probabilidade_suporte': support_proba
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