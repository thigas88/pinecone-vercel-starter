from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
import pickle
import os
import json
from sqlalchemy.dialects.postgresql import JSONB
from typing import Dict, List, Any, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import io
import re
import string
import unicodedata # Para remover acentos
import matplotlib.pyplot as plt
from collections import deque
import time
import threading
import sqlite3
import logging
from datetime import datetime
import tempfile
import aiofiles
from pathlib import Path

from config import logger
from monitor import ConceptDriftDetector

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from models import Base, Document, IngestHistory, DocumentChunk
from typing import List
from pydantic import BaseModel, ConfigDict

from dotenv import find_dotenv, load_dotenv

# Import new modules
from document_processor import DocumentProcessor, SplittingConfig
from vector_store import VectorStoreManager

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://chatrag:chatrag@localhost:5432/chatrag')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Cria as tabelas se não existirem
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="API de ChatRAG",
    description="API para disponibilização de funcionalidades",
    version="1.0.0"
)

# Dependency para obter sessão do banco
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models for document ingestion
class DocumentIngestRequest(BaseModel):
    """Request model for document ingestion with complete content."""
    content: Optional[str] = Field(None, description="Document content as text")
    url: Optional[str] = Field(None, description="URL of the document source")
    title: str = Field(..., description="Document title")
    category: Optional[str] = Field(None, description="Document category")
    tags: Optional[List[str]] = Field(default_factory=list, description="Document tags")
    keywords: Optional[List[str]] = Field(default_factory=list, description="Document keywords")
    
    # Splitting configuration
    splitting_method: str = Field("character", description="Splitting method: character, sentence, semantic, markdown")
    chunk_size: int = Field(1000, description="Target chunk size in characters")
    chunk_overlap: int = Field(200, description="Overlap between chunks")
    semantic_threshold: float = Field(0.5, description="Similarity threshold for semantic splitting")
    
    # File metadata
    file_type: Optional[str] = Field(None, description="File type: pdf, doc, web, markdown")
    
    @validator('splitting_method')
    def validate_splitting_method(cls, v):
        valid_methods = ["character", "sentence", "semantic", "markdown"]
        if v not in valid_methods:
            raise ValueError(f"Invalid splitting method. Must be one of: {valid_methods}")
        return v
    
    @validator('chunk_size')
    def validate_chunk_size(cls, v):
        if v < 100 or v > 10000:
            raise ValueError("Chunk size must be between 100 and 10000")
        return v

class DocumentIngestResponse(BaseModel):
    """Response model for document ingestion."""
    document_id: int
    status: str
    message: str
    chunks_created: int
    ingest_history_id: int

# Pydantic para entrada/saída
class DocumentCreate(BaseModel):
    url: str
    title: Optional[str] = None
    tags: Optional[str] = None
    category: Optional[str] = None
    splitting_method: Optional[str] = None
    chunk_size: Optional[int] = None
    overlap: Optional[int] = None
    file_name: Optional[str] = None
    scheduled_at: Optional[str] = None  # ISO8601 string

class DocumentOut(BaseModel):
    id: int
    url: str
    title: Optional[str] = None
    tags: Optional[str] = None
    category: Optional[str] = None
    splitting_method: Optional[str] = None
    chunk_size: Optional[int] = None
    overlap: Optional[int] = None
    file_name: Optional[str] = None
    status: Optional[str] = 'agendado'
    
    # Use 'datetime' para os campos de data. FastAPI cuidará da serialização.
    created_at: datetime
    updated_at: datetime
    
    # Use 'Optional[datetime]' para datas que podem ser nulas.
    scheduled_at: Optional[datetime] = None
    last_indexed_at: Optional[datetime] = None

    # Configuração importante para permitir que o Pydantic leia os dados
    # diretamente de um objeto ORM (como seus modelos SQLAlchemy).
    # Em versões mais antigas do Pydantic, isso era feito com 'class Config: orm_mode = True'
    model_config = ConfigDict(from_attributes=True)


class IngestHistoryOut(BaseModel):
    id: int
    document_id: int
    started_at: datetime
    finished_at: Optional[datetime] = None
    status: str
    message: Optional[str] = None
    chunks_indexed: Optional[int] = None
    error: Optional[str] = None
    model_config = ConfigDict(from_attributes=True)

# Initialize document processor and vector store
document_processor = DocumentProcessor()
vector_store_manager = VectorStoreManager()

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
    confianca_categoria: Optional[Dict[str, float]] = None
    confianca_suporte: Optional[Union[Dict[int, float], float]] = None

# Modelo de dados para resposta de treinamento
class TrainingResponse(BaseModel):
    message: str
    accuracy_categoria: float
    accuracy_suporte: float
    classification_report_categoria: Dict[str, Any]
    classification_report_suporte: Dict[str, Any]
    best_params_categoria: Dict[str, Any]
    best_params_suporte: Dict[str, Any]

class ChunkIn(BaseModel):
    chunk_index: int
    chunk_text: str
    hash: str = None

class ChunksIngestRequest(BaseModel):
    ingest_history_id: int = None  # opcional, se quiser associar ao histórico
    chunks: List[ChunkIn]

class ChunkOut(BaseModel):
    id: int
    document_id: int
    ingest_history_id: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None
    chunk_index: int
    chunk_text: str
    hash: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

def remove_accents(text):
        """Remove acentos de um texto."""
        return ''.join(c for c in unicodedata.normalize('NFD', text)
                      if unicodedata.category(c) != 'Mn')

def preprocess_text(text: str) -> str:
    """
    Realiza pré-processamento básico no texto:
    1. Converte para minúsculas.
    2. Remove acentos.
    3. Remove pontuações.
    4. Remove espaços extras.
    """
    # 1. Minúsculas
    text = text.lower()

    # 2. Remover acentos
    # nfkd_form = unicodedata.normalize('NFD', text)
    # text = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    text = remove_accents(text)

    # 3. Remover caracteres especiais
    text = re.sub(r'[^\w\s]', ' ', text)

    # 4. Remover pontuações
    # Usar str.maketrans para eficiência
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # 5. Remover espaços extras (no início/fim e múltiplos espaços internos)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

    # Tokenização
    tokens = word_tokenize(text)
    
    # Remover stopwords e aplicar stemming
    tokens = [stemmer.stem(word) for word in tokens if word not in stopwords and len(word) > 2]
    
    return ' '.join(tokens)


@app.get("/health")
async def health_check():
    """
    Verifica se a API está funcionando.
    
    Retorna o status da API.
    """
    return {"status": "OK"}

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
            ('tfidf', TfidfVectorizer(
                preprocessor=preprocess_text,
                max_features=10000
            )),
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
                'classifier': [LinearSVC()],
                'classifier__C': [0.1, 1.0, 10.0],
            },
            {
                'tfidf__max_features': [None, 5000, 10000],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'classifier': [SVC()],
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__kernel': ['sigmoid'],
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
        if isinstance(best_model_cat.named_steps['classifier'], SVC) or isinstance(best_model_cat.named_steps['classifier'], LinearSVC):
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



        # Inicializar detector de concept drift
        drift_detector = ConceptDriftDetector(
            window_size=100,
            threshold=0.05,
            metrics=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        )
        drift_detector.initialize_db()
        
        # Feedback loop e retreinamento
        feedback_data = []
        last_retrain_time = time.time()
        retrain_interval = 86400  # 24 horas em segundos
        min_feedback_for_retrain = 50
        monitoring_active = False

        # Configurar o detector de concept drift com os resultados do conjunto de teste
        drift_detector.set_reference_performance(y_test_cat, y_pred_cat)
        
        # Salvar dados de teste para validação contínua
        test_data = pd.DataFrame({
            'text': X_test_cat,
            'category': y_test_cat
        })
        test_data.to_csv('test_data.csv', index=False)
        

        
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
        
        question = preprocess_text(question_request.pergunta)
        
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

        print(f"Pergunta: {question}")
        print(f"Categoria: {category}")
        print(f"Suporte Técnico: {support_bool}")
        print(f"Probabilidades Categoria: {cat_proba_dict}")
        print(f"Probabilidade Suporte: {support_proba}")


        try:
            # Primeiro vamos transformar o texto usando o TF-IDF
            X_cat_transformed = category_model.named_steps['tfidf'].transform([question])
            
            # Agora podemos acessar o estimador com os dados transformados
            if isinstance(category_model.named_steps['classifier'], CalibratedClassifierCV):
                # Para CalibratedClassifierCV, acessamos o estimador base
                estimator = category_model.named_steps['classifier'].calibrated_classifiers_[0].estimator
                
                # Verificar que tipo de estimador é
                if isinstance(estimator, LinearSVC) or isinstance(estimator, SVC):
                    category_decision_scores = estimator.decision_function(X_cat_transformed)
                else:
                    # Usar predict_proba para outros tipos de estimadores
                    category_decision_scores = category_model.predict_proba([question])
                    category_confidence = {
                        cls: float(category_decision_scores[0][i]) for i, cls in enumerate(category_model.classes_)
                    }
                    # Pular o resto do cálculo de confiança
                    category_sorted_confidence = dict(sorted(category_confidence.items(), key=lambda x: x[1], reverse=True))
                    # Ir direto para o return
                    print(f"Usando probabilidades diretamente: {category_sorted_confidence}")
            else:
                # Se não for calibrado, tentar usar decision_function diretamente
                category_decision_scores = category_model.named_steps['classifier'].decision_function(X_cat_transformed)
            
            
            category_confidence = {}
            category_classes = category_model.classes_

            print(f"Scores de decisão para categoria: {category_decision_scores}")
            print(f"Shape dos scores de categoria: {category_decision_scores.shape}")
            
    
            # Para classificador multiclasse, temos um score para cada classe
            scores = category_decision_scores[0]  # Primeiro elemento é um array de scores
            category_confidence = {
                cls: float(scores[i]) for i, cls in enumerate(category_classes)
            }
            
            # Normalizar os scores para obter probabilidades
            category_max_abs_score = max([abs(score) for score in category_confidence.values()])
            if category_max_abs_score > 0:  # Evitar divisão por zero
                category_normalized_confidence = {k: (v / category_max_abs_score + 1) / 2 for k, v in category_confidence.items()}
            else:
                # Se todos os scores forem zero, distribuir igualmente
                category_normalized_confidence = {k: 1.0/len(category_confidence) for k in category_confidence}
            
            category_total = sum(category_normalized_confidence.values())
            category_normalized_confidence = {k: v / category_total for k, v in category_normalized_confidence.items()}
            
            # Ordenar por confiança
            category_sorted_confidence = dict(sorted(category_normalized_confidence.items(), key=lambda x: x[1], reverse=True))
            

        except Exception as e:
            print(f"Erro ao obter scores de decisão para categoria: {e}")
            # Fallback - use as probabilidades se disponíveis
            category_sorted_confidence = cat_proba_dict
            print("Usando probabilidades (proba) para categoria como fallback")
        
        
        print(f"Confiança Categoria: {category_sorted_confidence}")
        

        # Obter probabilidade de suporte técnico - classificador binário
        # Para classificador binário, o score é um único valor
        try:

            support_decision_scores = support_model.predict_proba([question])
            support_confidence = {}
            support_classes = support_model.classes_
            print(f"Classes de suporte: {support_classes}")
            print(f"Scores de decisão para suporte: {support_decision_scores}")
            print(f"Shape dos scores de suporte: {support_decision_scores.shape}")
            

            # Para classificador binário, o score é um único valor
            score_value = support_decision_scores[0]  # Primeiro elemento do array
            support_confidence = {
                support_classes[0]: score_value[0],  # Classe negativa
                support_classes[1]: score_value[1]    # Classe positiva
            }

            
            
        except Exception as e:
            print(f"Erro ao obter scores de decisão para suporte: {e}")
            # Fallback - use as probabilidades se disponíveis
            support_confidence = support_proba
            print("Usando probabilidades (proba) para suporte como fallback")
        
        
        print(f"Confiança Suporte: {support_confidence}")

        
        return {
            'pergunta': question,
            'categoria': category,
            'suporte_tecnico': support_bool,
            'probabilidades_categoria': cat_proba_dict,
            'confianca_categoria': category_sorted_confidence,
            'confianca_suporte': support_confidence,
            'probabilidade_suporte': support_proba
        }
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/admin/documents/{document_id}/chunks')
def add_chunks(document_id: int, req: ChunksIngestRequest, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail='Documento não encontrado')
    for chunk in req.chunks:
        db_chunk = DocumentChunk(
            document_id=document_id,
            ingest_history_id=req.ingest_history_id,
            chunk_index=chunk.chunk_index,
            chunk_text=chunk.chunk_text,
            hash=chunk.hash
        )
        db.add(db_chunk)
    db.commit()
    return {"message": f"{len(req.chunks)} chunks adicionados ao documento {document_id}."}

@app.get('/admin/documents/{document_id}/chunks', response_model=List[ChunkOut])
def get_chunks(document_id: int, db: Session = Depends(get_db)):
    chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).order_by(DocumentChunk.chunk_index).all()
    return chunks

@app.post('/admin/documents', response_model=DocumentOut)
def create_document(doc: DocumentCreate, db: Session = Depends(get_db)):
    db_doc = Document(
        url=doc.url,
        title=doc.title,
        tags=doc.tags,
        category=doc.category,
        splitting_method=doc.splitting_method,
        chunk_size=doc.chunk_size,
        overlap=doc.overlap,
        file_name=doc.file_name,
        scheduled_at=doc.scheduled_at
    )
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)
    return db_doc

@app.get('/admin/documents', response_model=List[DocumentOut])
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    print(docs)
    return docs

@app.get('/admin/documents/{doc_id}/history', response_model=List[IngestHistoryOut])
def get_document_history(doc_id: int, db: Session = Depends(get_db)):
    history = db.query(IngestHistory).filter(IngestHistory.document_id == doc_id).order_by(IngestHistory.started_at.desc()).all()
    return history

@app.post('/admin/documents/{doc_id}/reindex')
def reindex_document(doc_id: int, db: Session = Depends(get_db)):
    # Aqui entraria a lógica de reindexação: apagar chunks antigos do Pinecone, reindexar, atualizar status e histórico
    # Exemplo de atualização de status:
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail='Documento não encontrado')
    doc.status = 'processando'
    db.commit()
    # Chamar rotina de reindexação (assíncrona ou não)
    # Após finalizar, atualizar status e inserir registro em ingest_history
    return {"message": f"Reindexação do documento {doc_id} agendada/iniciada."}

@app.post('/admin/ingest/document', response_model=DocumentIngestResponse)
async def ingest_document(
    request: DocumentIngestRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Ingest a complete document with automatic text splitting and vector storage.
    
    This endpoint accepts a complete document and processes it according to the
    specified splitting parameters. The document is split into chunks, enriched
    with metadata, and stored in the vector database.
    """
    try:
        # Create document record
        db_doc = Document(
            url=request.url,
            title=request.title,
            tags=",".join(request.tags) if request.tags else None,
            category=request.category,
            splitting_method=request.splitting_method,
            chunk_size=request.chunk_size,
            overlap=request.chunk_overlap,
            file_name=None,
            status="processing"
        )
        db.add(db_doc)
        db.commit()
        db.refresh(db_doc)
        
        # Create ingest history record
        ingest_history = IngestHistory(
            document_id=db_doc.id,
            status="processing",
            message="Document ingestion started"
        )
        db.add(ingest_history)
        db.commit()
        db.refresh(ingest_history)
        
        # Process document in background
        background_tasks.add_task(
            process_document_async,
            db_doc.id,
            ingest_history.id,
            request,
            db
        )
        
        return DocumentIngestResponse(
            document_id=db_doc.id,
            status="processing",
            message="Document ingestion started",
            chunks_created=0,
            ingest_history_id=ingest_history.id
        )
        
    except Exception as e:
        logger.error(f"Failed to start document ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/admin/ingest/file', response_model=DocumentIngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    title: str = None,
    category: str = None,
    tags: List[str] = [],
    splitting_method: str = "character",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    semantic_threshold: float = 0.5,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Ingest a file (PDF, DOCX, TXT, HTML, MD) with automatic text splitting.
    
    Upload a file and it will be processed according to the specified splitting
    parameters. Supported formats: PDF, DOCX, TXT, HTML, Markdown.
    """
    try:
        # Validate file type
        file_extension = Path(file.filename).suffix.lower()
        supported_types = {'.pdf': 'pdf', '.docx': 'docx', '.txt': 'txt', 
                          '.html': 'html', '.md': 'markdown', '.markdown': 'markdown'}
        
        if file_extension not in supported_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Supported: {list(supported_types.keys())}"
            )
        
        file_type = supported_types[file_extension]
        
        # Save uploaded file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        
        # Create document record
        db_doc = Document(
            url=f"file://{file.filename}",
            title=title or file.filename,
            tags=",".join(tags) if tags else None,
            category=category,
            splitting_method=splitting_method,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            file_name=file.filename,
            status="processing"
        )
        db.add(db_doc)
        db.commit()
        db.refresh(db_doc)
        
        # Create ingest history record
        ingest_history = IngestHistory(
            document_id=db_doc.id,
            status="processing",
            message="File ingestion started"
        )
        db.add(ingest_history)
        db.commit()
        db.refresh(ingest_history)
        
        # Process file in background
        background_tasks.add_task(
            process_file_async,
            db_doc.id,
            ingest_history.id,
            temp_file.name,
            file_type,
            {
                "title": title or file.filename,
                "category": category,
                "tags": tags,
                "splitting_method": splitting_method,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "semantic_threshold": semantic_threshold
            },
            db
        )
        
        return DocumentIngestResponse(
            document_id=db_doc.id,
            status="processing",
            message="File ingestion started",
            chunks_created=0,
            ingest_history_id=ingest_history.id
        )
        
    except Exception as e:
        logger.error(f"Failed to start file ingestion: {e}")
        if hasattr(e, 'status_code'):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


async def process_document_async(
    document_id: int,
    ingest_history_id: int,
    request: DocumentIngestRequest,
    db: Session
):
    """Process document asynchronously."""
    try:
        # Create splitting configuration
        config = SplittingConfig(
            method=request.splitting_method,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            semantic_threshold=request.semantic_threshold
        )
        
        # Prepare metadata
        metadata = {
            "title": request.title,
            "category": request.category,
            "tags": request.tags,
            "keywords": request.keywords,
            "url": request.url,
            "type": request.file_type or "text"
        }
        
        # Process document
        chunks = document_processor.process_document(
            content=request.content,
            config=config,
            metadata=metadata
        )
        
        # Store chunks in database
        for chunk in chunks:
            db_chunk = DocumentChunk(
                document_id=document_id,
                ingest_history_id=ingest_history_id,
                meta=metadata,
                chunk_index=chunk["index"],
                chunk_text=chunk["text"],
                hash=chunk["hash"]
            )
            db.add(db_chunk)
        
        db.commit()
        
        # Add chunks to vector store
        chunk_ids = vector_store_manager.add_documents(
            chunks=chunks,
            document_id=document_id
        )
        
        # Update document status
        doc = db.query(Document).filter(Document.id == document_id).first()
        doc.status = "indexed"
        doc.last_indexed_at = datetime.utcnow()
        
        # Update ingest history
        history = db.query(IngestHistory).filter(IngestHistory.id == ingest_history_id).first()
        history.status = "completed"
        history.finished_at = datetime.utcnow()
        history.chunks_indexed = len(chunks)
        history.message = f"Successfully indexed {len(chunks)} chunks"
        
        db.commit()
        
        logger.info(f"Document {document_id} processed successfully with {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Failed to process document {document_id}: {e}")
        
        # Update status on error
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.status = "error"
        
        history = db.query(IngestHistory).filter(IngestHistory.id == ingest_history_id).first()
        if history:
            history.status = "error"
            history.finished_at = datetime.utcnow()
            history.error = str(e)
        
        db.commit()

async def process_file_async(
    document_id: int,
    ingest_history_id: int,
    file_path: str,
    file_type: str,
    params: Dict[str, Any],
    db: Session
):
    """Process file asynchronously."""
    try:
        # Create splitting configuration
        config = SplittingConfig(
            method=params["splitting_method"],
            chunk_size=params["chunk_size"],
            chunk_overlap=params["chunk_overlap"],
            semantic_threshold=params["semantic_threshold"]
        )
        
        # Prepare metadata
        metadata = {
            "title": params["title"],
            "category": params["category"],
            "tags": params["tags"],
            "type": file_type
        }
        
        # Process file
        chunks = document_processor.process_file(
            file_path=file_path,
            file_type=file_type,
            config=config,
            metadata=metadata
        )
        
        # Store chunks in database
        for chunk in chunks:
            db_chunk = DocumentChunk(
                document_id=document_id,
                ingest_history_id=ingest_history_id,
                chunk_index=chunk["index"],
                chunk_text=chunk["text"],
                hash=chunk["hash"]
            )
            db.add(db_chunk)
        
        db.commit()
        
        # Add chunks to vector store
        chunk_ids = vector_store_manager.add_documents(
            chunks=chunks,
            document_id=document_id
        )
        
        # Update document status
        doc = db.query(Document).filter(Document.id == document_id).first()
        doc.status = "indexed"
        doc.last_indexed_at = datetime.utcnow()
        
        # Update ingest history
        history = db.query(IngestHistory).filter(IngestHistory.id == ingest_history_id).first()
        history.status = "completed"
        history.finished_at = datetime.utcnow()
        history.chunks_indexed = len(chunks)
        history.message = f"Successfully indexed {len(chunks)} chunks"
        
        db.commit()
        
        # Clean up temporary file
        os.unlink(file_path)
        
        logger.info(f"File {document_id} processed successfully with {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Failed to process file {document_id}: {e}")
        
        # Update status on error
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.status = "error"
        
        history = db.query(IngestHistory).filter(IngestHistory.id == ingest_history_id).first()
        if history:
            history.status = "error"
            history.finished_at = datetime.utcnow()
            history.error = str(e)
        
        db.commit()
        
        # Clean up temporary file
        if os.path.exists(file_path):
            os.unlink(file_path)

@app.get('/search')
async def search_documents(
    query: str,
    k: int = 5,
    category: Optional[str] = None,
    tags: Optional[str] = None,
    score_threshold: Optional[float] = None
):
    """
    Search for documents using vector similarity.
    
    Args:
        query: Search query
        k: Number of results to return
        category: Filter by category
        tags: Filter by tags separed by comman
        score_threshold: Minimum similarity score
    """
    try:
        # Build filters
        filters = {}
        if category:
            filters["category"] = category
        if tags:
            # Converte a string de tags separadas por vírgula em uma lista,
            # removendo espaços em branco extras de cada tag.
            processed_tags = [tag.strip() for tag in tags.split(',')]
            filters["tags"] = processed_tags
        
        # Perform search
        results = vector_store_manager.similarity_search(
            query=query,
            k=k,
            filter_dict=filters,
            score_threshold=score_threshold
        )
        
        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": 1 - float(abs(score)) # Reverse the sign of the number to calculate the distance from 1 
            })
        
        return {
            "query": query,
            "results": formatted_results,
            "count": len(formatted_results)
        }
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/admin/vector-store/stats')
async def get_vector_store_stats():
    """Get statistics about the vector store."""
    try:
        stats = vector_store_manager.get_collection_stats()
        print(stats)
        return stats
    except Exception as e:
        logger.error(f"Failed to get vector store stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)