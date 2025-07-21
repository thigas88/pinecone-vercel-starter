import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import threading
import sqlite3
import logging
import json
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score
import datetime
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

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
        
    def initialize_db(self, db_path="performance_monitoring.db"):
        """Inicializa o banco de dados para armazenar métricas de desempenho"""
        try:
            self.db_conn = sqlite3.connect(db_path)
            cursor = self.db_conn.cursor()
            
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
            
            self.db_conn.commit()
            logger.info(f"Banco de dados de monitoramento inicializado em {db_path}")
        except Exception as e:
            logger.error(f"Erro ao inicializar banco de dados: {e}")
    
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
        
        # Salvar no banco de dados
        if self.db_conn:
            cursor = self.db_conn.cursor()
            for metric, value in self.reference_performance.items():
                cursor.execute(
                    "INSERT INTO performance_metrics (timestamp, window_id, metric, value) VALUES (?, ?, ?, ?)",
                    (datetime.datetime.now().isoformat(), 0, metric, value)
                )
            self.db_conn.commit()
    
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
        
        # Salvar predição no banco de dados
        if self.db_conn:
            cursor = self.db_conn.cursor()
            cursor.execute(
                "INSERT INTO predictions (timestamp, text, true_label, predicted_label, correct) VALUES (?, ?, ?, ?, ?)",
                (datetime.datetime.now().isoformat(), text, true_label, predicted_label, correct)
            )
            self.db_conn.commit()
        
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
        
        # Salvar no banco de dados
        if self.db_conn:
            cursor = self.db_conn.cursor()
            for metric, value in self.current_performance.items():
                cursor.execute(
                    "INSERT INTO performance_metrics (timestamp, window_id, metric, value) VALUES (?, ?, ?, ?)",
                    (datetime.datetime.now().isoformat(), window_id, metric, value)
                )
            self.db_conn.commit()
        
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
                
                # Salvar alerta no banco de dados
                if self.db_conn:
                    cursor = self.db_conn.cursor()
                    cursor.execute(
                        "INSERT INTO drift_alerts (timestamp, metric, reference_value, current_value, difference) VALUES (?, ?, ?, ?, ?)",
                        (drift_alert['timestamp'], metric, ref_value, current_value, difference)
                    )
                    self.db_conn.commit()
                
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
        logger.info(f"Gráfico de tendência salvo em {filepath}")

