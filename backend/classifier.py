import pandas as pd
import numpy as np
import re
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer  # Stemmer para português
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.base import clone
import joblib
import logging
import unicodedata
import datetime
import json
import os
import matplotlib
matplotlib.use('Agg')  # Usar backend não interativo
import matplotlib.pyplot as plt
from collections import deque
import time
import threading
import sqlite3

from config import logger


class ConceptDriftDetector:
    """Classe para detectar concept drift em modelos de classificação"""
    
    def __init__(self, window_size=100, threshold=0.05, metrics=['accuracy', 'f1_macro']):
        """
        Inicializa o detector de concept drift
        
        Args:
            window_size: Número de predições para considerar na janela deslizante
            threshold: Limiar de diferença de desempenho que indica concept drift
            metrics: Lista de métricas para monitorar
        """
        self.window_size = window_size
        self.threshold = threshold
        self.metrics = metrics
        self.predictions = deque(maxlen=window_size)
        self.reference_performance = {}
        self.current_performance = {}
        self.performance_history = []
        self.drift_alerts = []
        self.db_conn = None
        self.db_path = None  # Armazenar o caminho do banco para criar novas conexões quando necessário

        
    def initialize_db(self, db_path="performance_monitoring.db"):
        """Inicializa o banco de dados para armazenar métricas de desempenho"""
        try:
            self.db_path = db_path  # Armazenar o caminho para futuras conexões
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Tabela para métricas de desempenho
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                window_id INTEGER,
                metric TEXT,
                value REAL
            )
            ''')
            
            # Tabela para alertas de concept drift
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS drift_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                metric TEXT,
                reference_value REAL,
                current_value REAL,
                difference REAL
            )
            ''')
            
            # Tabela para predições individuais
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                text TEXT,
                true_label TEXT,
                predicted_label TEXT,
                correct INTEGER
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"Banco de dados de monitoramento inicializado em {db_path}")
        except Exception as e:
            logger.error(f"Erro ao inicializar banco de dados: {e}")
    
    def _get_db_connection(self):
        """Retorna uma nova conexão com o banco de dados para a thread atual"""
        if self.db_path:
            try:
                return sqlite3.connect(self.db_path)
            except Exception as e:
                logger.error(f"Erro ao conectar ao banco de dados: {e}")
                return None
        return None
    
    def close_db(self):
        """Fecha a conexão com o banco de dados"""
        if self.db_conn:
            self.db_conn.close()
    
    def set_reference_performance(self, y_true, y_pred):
        """Define o desempenho de referência inicial baseado nos resultados do conjunto de teste"""
        for metric in self.metrics:
            if metric == 'accuracy':
                self.reference_performance[metric] = accuracy_score(y_true, y_pred)
            elif metric == 'f1_macro':
                self.reference_performance[metric] = f1_score(y_true, y_pred, average='macro')
            elif metric == 'precision_macro':
                self.reference_performance[metric] = precision_score(y_true, y_pred, average='macro')
            elif metric == 'recall_macro':
                self.reference_performance[metric] = recall_score(y_true, y_pred, average='macro')
                
        logger.info(f"Desempenho de referência definido: {self.reference_performance}")
        
        # Salvar no histórico
        self.performance_history.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'window_id': 0,
            'metrics': self.reference_performance.copy()
        })
        
        # Salvar no banco de dados com uma nova conexão
        conn = self._get_db_connection()
        if conn:
            cursor = conn.cursor()
            for metric, value in self.reference_performance.items():
                cursor.execute(
                    "INSERT INTO performance_metrics (timestamp, window_id, metric, value) VALUES (?, ?, ?, ?)",
                    (datetime.datetime.now().isoformat(), 0, metric, value)
                )
            conn.commit()
            conn.close()
    
    def add_prediction(self, text, true_label, predicted_label):
        """Adiciona uma nova predição à janela de monitoramento"""
        correct = 1 if true_label == predicted_label else 0
        self.predictions.append({
            'text': text,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'correct': correct,
            'timestamp': datetime.datetime.now().isoformat()
        })
        
        # Salvar predição no banco de dados com uma nova conexão
        conn = self._get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO predictions (timestamp, text, true_label, predicted_label, correct) VALUES (?, ?, ?, ?, ?)",
                    (datetime.datetime.now().isoformat(), text, true_label, predicted_label, correct)
                )
                conn.commit()
            finally:
                conn.close()
        
        # Se a janela estiver cheia, avaliar o desempenho atual
        if len(self.predictions) == self.window_size:
            self.evaluate_current_performance()
    
    def evaluate_current_performance(self):
        """Avalia o desempenho atual com base na janela de predições"""
        if len(self.predictions) < self.window_size // 2:
            return  # Não avaliar se não tivermos dados suficientes
            
        y_true = [p['true_label'] for p in self.predictions]
        y_pred = [p['predicted_label'] for p in self.predictions]
        
        window_id = len(self.performance_history)
        
        self.current_performance = {}
        for metric in self.metrics:
            if metric == 'accuracy':
                self.current_performance[metric] = accuracy_score(y_true, y_pred)
            elif metric == 'f1_macro':
                self.current_performance[metric] = f1_score(y_true, y_pred, average='macro')
            elif metric == 'precision_macro':
                self.current_performance[metric] = precision_score(y_true, y_pred, average='macro')
            elif metric == 'recall_macro':
                self.current_performance[metric] = recall_score(y_true, y_pred, average='macro')
        
        # Salvar no histórico
        self.performance_history.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'window_id': window_id,
            'metrics': self.current_performance.copy()
        })
        
        # Salvar no banco de dados com uma nova conexão
        conn = self._get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                for metric, value in self.current_performance.items():
                    cursor.execute(
                        "INSERT INTO performance_metrics (timestamp, window_id, metric, value) VALUES (?, ?, ?, ?)",
                        (datetime.datetime.now().isoformat(), window_id, metric, value)
                    )
                conn.commit()
            finally:
                conn.close()
        
        # Verificar se há concept drift
        self.check_for_drift()
    
    def check_for_drift(self):
        """Verifica se há evidência de concept drift comparando o desempenho atual com o de referência"""
        for metric in self.metrics:
            ref_value = self.reference_performance.get(metric, 0)
            current_value = self.current_performance.get(metric, 0)
            
            # Calcula a diferença absoluta
            difference = abs(ref_value - current_value)
            
            # Verifica se a diferença ultrapassa o limiar
            if difference > self.threshold:
                drift_alert = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'metric': metric,
                    'reference_value': ref_value,
                    'current_value': current_value,
                    'difference': difference
                }
                
                self.drift_alerts.append(drift_alert)
                
                # Salvar alerta no banco de dados com uma nova conexão
                conn = self._get_db_connection()
                if conn:
                    try:
                        cursor = conn.cursor()
                        cursor.execute(
                            "INSERT INTO drift_alerts (timestamp, metric, reference_value, current_value, difference) VALUES (?, ?, ?, ?, ?)",
                            (drift_alert['timestamp'], metric, ref_value, current_value, difference)
                        )
                        conn.commit()
                    finally:
                        conn.close()
                
                logger.warning(f"CONCEPT DRIFT DETECTADO! Métrica: {metric}, "
                            f"Referência: {ref_value:.4f}, Atual: {current_value:.4f}, "
                            f"Diferença: {difference:.4f}")
    
    def export_metrics(self, filepath="performance_metrics.json"):
        """Exporta as métricas de desempenho para um arquivo JSON"""
        data = {
            'reference_performance': self.reference_performance,
            'performance_history': self.performance_history,
            'drift_alerts': self.drift_alerts
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Métricas exportadas para {filepath}")
    
    def plot_performance_trend(self, metric='accuracy', filepath="performance_trend.png"):
        """Gera um gráfico da tendência de desempenho para uma métrica específica"""
        if not self.performance_history:
            logger.warning("Sem dados de histórico para gerar gráfico")
            return
            
        windows = [entry['window_id'] for entry in self.performance_history]
        values = [entry['metrics'].get(metric, 0) for entry in self.performance_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(windows, values, marker='o', linestyle='-')
        plt.axhline(y=self.reference_performance.get(metric, 0), color='r', linestyle='--', 
                   label=f'Referência: {self.reference_performance.get(metric, 0):.4f}')
        
        # Adicionar linhas verticais para alertas de drift
        for alert in self.drift_alerts:
            if alert['metric'] == metric:
                alert_window = next((i for i, entry in enumerate(self.performance_history) 
                                   if entry['timestamp'] == alert['timestamp']), None)
                if alert_window is not None:
                    plt.axvline(x=alert_window, color='red', alpha=0.5)
        
        plt.title(f'Tendência de Desempenho - {metric}')
        plt.xlabel('Janela de Avaliação')
        plt.ylabel(f'Valor de {metric}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()  # Fechar a figura explicitamente
        logger.info(f"Gráfico de tendência salvo em {filepath}")


class CategoryClassifier:
    def __init__(self):
        # Definição das categorias e palavras-chave associadas
        self.categories = {
            'ecampus': ['ecampus', 'e-campus', 'nota', 'disciplina', 'matrícula', 'professor', 'aluno', 'turma', 'curso'],
            'sei': ['sei', 'processo', 'documento', 'protocolo', 'assinatura eletrônica', 'peticionamento'],
            'contaInstitucional': ['email', 'e-mail', 'conta', 'senha', 'login', 'acesso', 'redefinir', 'institucional', 'alterar senha'],
            'glpi': ['chamado', 'ticket', 'suporte', 'problema técnico', 'não consigo acessar', 'erro', 'bug'],
            'revista': ['revista', 'publicar artigo', 'criar artigo', 'artigo', 'artigos', 'publicação', 'publicações', 'portal revistas'],
        }
        
        # Baixar recursos do NLTK se necessário
        nltk.download('rslp')
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            
        self.stopwords = set(stopwords.words('portuguese'))
        self.stemmer = RSLPStemmer()
        self.model = None
        self.vectorizer = None
        
        # Inicializar detector de concept drift
        self.drift_detector = ConceptDriftDetector(
            window_size=100,
            threshold=0.05,
            metrics=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        )
        self.drift_detector.initialize_db()
        
        # Feedback loop e retreinamento
        self.feedback_data = []
        self.last_retrain_time = time.time()
        self.retrain_interval = 86400  # 24 horas em segundos
        self.min_feedback_for_retrain = 50
        self.monitoring_active = False
        
    def remove_accents(self, text):
        """Remove acentos de um texto."""
        return ''.join(c for c in unicodedata.normalize('NFD', text)
                      if unicodedata.category(c) != 'Mn')
        
    def preprocess_text(self, text):
        """Realiza todo o pré-processamento necessário no texto."""
        if not isinstance(text, str):
            return ""
            
        # Converter para minúsculas
        text = text.lower()
        
        # Remover acentos
        text = self.remove_accents(text)
        
        # Remover caracteres especiais
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenização
        tokens = word_tokenize(text)
        
        # Remover stopwords e aplicar stemming
        tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stopwords and len(word) > 2]
        
        return ' '.join(tokens)
    
    def generate_synthetic_data(self, samples_per_category=200):
        """Gera dados sintéticos para treinar o modelo."""
        data = []
        labels = []
        
        for category, keywords in self.categories.items():
            logger.info(f"Gerando dados para a categoria: {category}")
            
            # Palavras-chave básicas
            for keyword in keywords:
                data.append(keyword)
                labels.append(category)
            
            # Gerar frases de perguntas com as palavras-chave
            templates = [
                "Como faço para {}?",
                "Preciso de ajuda com {}.",
                "Estou com problema no {}.",
                "Não consigo acessar meu {}.",
                "Como posso {}?",
                "Gostaria de saber sobre {}.",
                "Tenho dúvidas sobre {}.",
                "{} não está funcionando.",
                "Como resolver problema de {}?",
                "Preciso de informações sobre {}."
            ]
            
            for _ in range(samples_per_category):
                # Escolher aleatoriamente uma ou mais palavras-chave
                num_keywords = np.random.randint(1, min(4, len(keywords)))
                selected_keywords = np.random.choice(keywords, num_keywords, replace=False)
                
                # Escolher um template aleatório
                template = np.random.choice(templates)
                
                # Preencher o template com as palavras-chave
                if '{}' in template:
                    question = template.format(' '.join(selected_keywords))
                else:
                    question = template + ' ' + ' '.join(selected_keywords)
                
                data.append(question)
                labels.append(category)
        
        return pd.DataFrame({'text': data, 'category': labels})
    
    def train(self, df=None, test_size=0.2):
        """Treina o modelo de classificação."""
        if df is None:
            logger.info("Gerando dados sintéticos para treinamento")
            df = self.generate_synthetic_data()
        
        logger.info(f"Total de exemplos: {len(df)}")
        
        # Pré-processar os textos
        logger.info("Pré-processando textos")
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Divisão entre treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], df['category'], test_size=test_size, random_state=42)
        
        # Pipeline de treinamento
        logger.info("Criando pipeline de treinamento")
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
            ('classifier', OneVsRestClassifier(LinearSVC()))
        ])
        
        # Hiperparâmetros para otimização
        parameters = {
            'tfidf__max_features': [5000, 10000],
            'tfidf__min_df': [1, 2],
            'classifier__estimator__C': [0.1, 1, 10],
        }
        
        # Busca em grade para encontrar os melhores hiperparâmetros
        logger.info("Iniciando GridSearchCV para otimização de hiperparâmetros")
        grid_search = GridSearchCV(pipeline, parameters, cv=5, verbose=1)
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Melhores parâmetros: {grid_search.best_params_}")
        self.model = grid_search.best_estimator_
        
        # Avaliação no conjunto de teste
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Acurácia no conjunto de teste: {accuracy:.4f}")
        logger.info("\nRelatório de classificação:")
        logger.info(classification_report(y_test, y_pred))
        
        # Configurar o detector de concept drift com os resultados do conjunto de teste
        self.drift_detector.set_reference_performance(y_test, y_pred)
        
        # Salvar dados de teste para validação contínua
        test_data = pd.DataFrame({
            'text': X_test,
            'category': y_test
        })
        test_data.to_csv('test_data.csv', index=False)
        
        return self.model
    
    def predict(self, question, true_label=None):
        """Prediz a categoria de uma pergunta."""
        if self.model is None:
            raise ValueError("Modelo não treinado. Execute o método 'train' primeiro.")
        
        processed_question = self.preprocess_text(question)
        question_array = np.array([processed_question])
        prediction = self.model.predict(question_array)[0]
        
        try:
            # Calcular confiança usando as distâncias da decisão
            decision_scores = self.model.decision_function([processed_question])[0]
            confidence = {}
            classes = self.model.classes_
            
            # Verificar formato da saída (binário vs multiclasse)
            if len(decision_scores.shape) == 1:
                # Caso binário
                for i, cls in enumerate(classes):
                    if i == 0:
                        confidence[cls] = float(-decision_scores[0])
                    else:
                        confidence[cls] = float(decision_scores[0])
            else:
                # Caso multiclasse
                for i, cls in enumerate(classes):
                    confidence[cls] = float(decision_scores[0][i])
            
            # Normalizar os scores para obter probabilidades
            max_abs_score = max([abs(score) for score in confidence.values()])
            if max_abs_score > 0:  # Evitar divisão por zero
                normalized_confidence = {k: (v / max_abs_score + 1) / 2 for k, v in confidence.items()}
            else:
                normalized_confidence = {k: 1.0/len(confidence) for k in confidence}
            
            total = sum(normalized_confidence.values())
            normalized_confidence = {k: v / total for k, v in normalized_confidence.items()}
        

            # Ordenar por confiança
            sorted_confidence = dict(sorted(normalized_confidence.items(), key=lambda x: x[1], reverse=True))
       
        except Exception as e:
            logger.warning(f"Erro ao calcular confiança: {e}. Usando probabilidades padrão.")
            # Fallback para probabilidades simples
            sorted_confidence = {cls: 1.0 if cls == prediction else 0.0 for cls in self.model.classes_}
    
        
 
        # Se temos um rótulo verdadeiro, adicioná-lo ao detector de concept drift
        if true_label is not None and self.monitoring_active:
            self.drift_detector.add_prediction(question, true_label, prediction)
            self.add_feedback(question, true_label, prediction)
        
        return {
            'category': prediction,
            'confidence': sorted_confidence
        }
    
    def add_feedback(self, question, true_label, predicted_label):
        """Adiciona feedback ao conjunto de dados para possível retreinamento"""
        self.feedback_data.append({
            'text': question,
            'category': true_label,
            'predicted': predicted_label,
            'timestamp': datetime.datetime.now().isoformat()
        })
        
        # Verificar se devemos retreinar
        self.check_for_retraining()
    
    def check_for_retraining(self):
        """Verifica se é hora de retreinar o modelo com base no feedback acumulado"""
        current_time = time.time()
        time_since_last_retrain = current_time - self.last_retrain_time
        
        # Se temos drift alerts ou suficiente feedback e passou tempo suficiente
        if ((len(self.drift_detector.drift_alerts) > 0 or 
             len(self.feedback_data) >= self.min_feedback_for_retrain) and
            time_since_last_retrain >= self.retrain_interval):
            
            logger.info("Iniciando processo de retreinamento automático")
            self.retrain_with_feedback()
    
    def retrain_with_feedback(self):
        """Retreina o modelo incorporando os dados de feedback"""
        if not self.feedback_data:
            logger.info("Sem dados de feedback para retreinar o modelo")
            return
            
        # Converter feedback para DataFrame
        feedback_df = pd.DataFrame(self.feedback_data)
        
        # Carregar dados de treinamento originais (sintéticos ou reais)
        try:
            original_df = pd.read_csv('training_data.csv')
        except FileNotFoundError:
            # Se não existir, gerar novos dados sintéticos
            original_df = self.generate_synthetic_data()
            
        # Combinar dados originais com feedback
        combined_df = pd.concat([original_df, feedback_df[['text', 'category']]], ignore_index=True)
        
        # Treinar novo modelo
        self.train(df=combined_df)
        
        # Atualizar timestamp do último retreinamento
        self.last_retrain_time = time.time()
        
        # Salvar dados de treinamento
        combined_df.to_csv('training_data.csv', index=False)
        
        # Salvar feedback separadamente para análise
        pd.DataFrame(self.feedback_data).to_csv('feedback_data.csv', index=False)
        
        # Limpar alertas de drift e dados de feedback
        self.drift_detector.drift_alerts = []
        self.feedback_data = []
        
        # Salvar o novo modelo
        self.save_model()
        
        logger.info("Modelo retreinado com sucesso incorporando dados de feedback")
    
    def start_monitoring(self):
        """Inicia o monitoramento de desempenho em uma thread separada"""
        self.monitoring_active = True
        
        # Iniciar thread para monitoramento contínuo
        monitor_thread = threading.Thread(target=self._continuous_monitoring)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        logger.info("Monitoramento de desempenho iniciado")
    
    def stop_monitoring(self):
        """Para o monitoramento de desempenho"""
        self.monitoring_active = False
        logger.info("Monitoramento de desempenho interrompido")
    
    def _continuous_monitoring(self):
        """Função executada em thread separada para monitoramento contínuo"""
        try:
            test_data = pd.read_csv('test_data.csv')
        except FileNotFoundError:
            logger.warning("Arquivo de teste não encontrado. Não é possível realizar monitoramento contínuo.")
            return
            
        interval = 3600  # 1 hora em segundos
        
        while self.monitoring_active:
            # A cada intervalo, verificar o desempenho em uma amostra dos dados de teste
            if test_data.shape[0] > 20:
                sample = test_data.sample(min(20, test_data.shape[0]))
                
                for _, row in sample.iterrows():
                    result = self.predict(row['text'])
                    self.drift_detector.add_prediction(row['text'], row['category'], result['category'])
            
            # Gerar relatórios e gráficos
            self.generate_performance_reports()
            
            # Dormir até o próximo intervalo
            time.sleep(interval)
    
    def generate_performance_reports(self):
        """Gera relatórios e gráficos de desempenho"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_dir = f"performance_reports_{timestamp}"
        
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
            
        # Exportar métricas
        self.drift_detector.export_metrics(f"{reports_dir}/metrics.json")
        
        # Gerar gráficos para cada métrica
        for metric in self.drift_detector.metrics:
            self.drift_detector.plot_performance_trend(
                metric=metric,
                filepath=f"{reports_dir}/{metric}_trend.png"
            )
            plt.close('all')  # Fechar todas as figuras para liberar recursos
            
        logger.info(f"Relatórios de desempenho gerados em {reports_dir}")
    
    def save_model(self, filepath="category_classifier_model.pkl"):
        """Salva o modelo treinado para uso futuro."""
        if self.model is None:
            raise ValueError("Não há modelo para salvar. Execute o método 'train' primeiro.")
        
        joblib.dump(self.model, filepath)
        logger.info(f"Modelo salvo em {filepath}")
        
    def load_model(self, filepath="category_classifier_model.pkl"):
        """Carrega um modelo previamente treinado."""
        try:
            self.model = joblib.load(filepath)
            logger.info(f"Modelo carregado de {filepath}")
        except Exception as e:
            logger.error(f"Erro ao carregar o modelo: {e}")
            raise

    def cleanup(self):
        """Realiza limpeza antes de finalizar"""
        self.stop_monitoring()
        self.drift_detector.close_db()
        logger.info("Recursos liberados e conexões fechadas")


# Exemplo de uso
if __name__ == "__main__":
    classifier = CategoryClassifier()
    model = classifier.train()
    classifier.save_model()
    
    # Iniciar monitoramento
    classifier.start_monitoring()
    
    # Testar com a pergunta de exemplo
    exemplo = "Como altero a senha da minha conta institucional?"
    resultado = classifier.predict(exemplo)
    print(f"\nPergunta: {exemplo}")
    print(f"Categoria prevista: {resultado['category']}")
    print("Confiança por categoria:")
    for cat, conf in resultado['confidence'].items():
        print(f"- {cat}: {conf:.4f}")
    
    # Exemplo de como fornecer feedback (simulando um usuário confirmando ou corrigindo)
    true_category = "contaInstitucional"  # Categoria correta fornecida pelo usuário
    classifier.predict(exemplo, true_label=true_category)
    
    # Em um cenário real, você executaria o sistema por mais tempo
    # Para demonstração, geramos alguns relatórios de desempenho
    classifier.generate_performance_reports()
    
    # Liberação de recursos
    classifier.cleanup()