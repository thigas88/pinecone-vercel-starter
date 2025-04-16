import requests
import json
import pandas as pd

# URL da API
API_URL = "http://localhost:8000"

def check_health():
    """Verifica se a API está funcionando."""
    response = requests.get(f"{API_URL}/health")
    print(f"Status da API: {response.json()}")

def train_model(file_path):
    """Treina o modelo com o arquivo CSV fornecido."""
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_URL}/train", files=files)
    
    if response.status_code == 200:
        print("Modelo treinado com sucesso!")
        print(f"Acurácia: {response.json().get('accuracy')}")
        print("\nMelhores parâmetros:")
        print(json.dumps(response.json().get('best_params'), indent=2))
    else:
        print(f"Erro ao treinar o modelo: {response.json().get('detail')}")

def classify_question(question):
    """Classifica uma pergunta utilizando o modelo treinado."""
    data = {'pergunta': question}
    response = requests.post(f"{API_URL}/classify", json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nPergunta: {result['pergunta']}")
        print(f"Categoria: {result['categoria']}")
        
        if result.get('probabilidades'):
            print("\nProbabilidades por categoria:")
            probas = result['probabilidades']
            # Ordenar por probabilidade em ordem decrescente
            sorted_probas = sorted(probas.items(), key=lambda x: x[1], reverse=True)
            for category, prob in sorted_probas:
                print(f"{category}: {prob:.4f}")
    else:
        print(f"Erro ao classificar a pergunta: {response.json().get('detail')}")

if __name__ == "__main__":
    # Verificar o status da API
    check_health()
    
    # Treinar o modelo
    train_model("data/dados_treinamento.csv")
    
    # Exemplos de classificação
    perguntas_exemplo = [
        "O que é assinatura digital e como ela se equipara à assinatura física?",
        "Qual o sistema utilizado para reunir as revistas eletrônicas no portal?",
        "Como faço para acessar o sistema de gerenciamento de wifi?",
        "Quais são os passos para acessar o SEI?",
        "Como posso votar no sistema de eleição eletrônica?"
    ]
    
    for pergunta in perguntas_exemplo:
        classify_question(pergunta)