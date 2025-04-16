# Classificador de Perguntas API

Esta API permite treinar um modelo de classificação de perguntas e utilizá-lo para classificar novas perguntas em categorias predefinidas.

## Categorias de Classificação

- assinatura
- ecampus
- pag
- sei
- revista
- wifi
- mautic
- metabase
- evoto

## Requisitos

```
fastapi==0.95.1
uvicorn==0.22.0
pandas==1.3.3
numpy==1.21.2
scikit-learn==1.0.1
python-multipart==0.0.6
pydantic==1.10.7
```

## Como Instalar

1. Clone este repositório
2. Instale as dependências:

```bash
pip install -r requirements.txt
```

## Como Executar

```bash
uvicorn app:app --reload
```

A API estará disponível em `http://localhost:8000`

## Documentação Interativa

Acesse a documentação interativa da API em:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`


## Docker

```sh
# Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build a imagem e execute

```sh
docker build -t meu-modelo .
docker run -p 8000:8000 meu-modelo
```

## Endpoints

### Verificar Status da API

```
GET /health
```

Resposta de sucesso:
```json
{
  "status": "OK"
}
```

### Treinar o Modelo

```
POST /train
```

Envie um arquivo CSV com as colunas 'pergunta' e 'tag' como um formulário multipart.

Exemplo de uso com curl:
```bash
curl -X POST -F "file=@dados_treinamento.csv" http://localhost:8000/train
```

Resposta de sucesso:
```json
{
  "message": "Modelo treinado e salvo com sucesso",
  "accuracy": 0.95,
  "classification_report": {...},
  "best_params": {...}
}
```

### Classificar uma Pergunta

```
POST /classify
```

Envie um JSON com a pergunta a ser classificada.

Exemplo de uso com curl:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"pergunta": "O que é assinatura digital?"}' http://localhost:8000/classify
```

Resposta de sucesso:
```json
{
  "pergunta": "O que é assinatura digital?",
  "categoria": "assinatura",
  "probabilidades": {
    "assinatura": 0.85,
    "ecampus": 0.05,
    "pag": 0.02,
    ...
  }
}
```

## Estrutura do Arquivo CSV de Treinamento

O arquivo CSV deve conter pelo menos as seguintes colunas:
- `pergunta`: O texto da pergunta a ser classificada
- `tag`: A categoria da pergunta (assinatura, ecampus, pag, sei, revista, wifi, mautic, metabase, evoto)

Exemplo:
```
pergunta,tag
"O que é assinatura digital?",assinatura
"Como faço para acessar o e-Campus?",ecampus
```

## Detalhes Técnicos

- A API utiliza FastAPI para criar endpoints rápidos e documentados automaticamente
- O modelo utiliza um pipeline de processamento de texto com TF-IDF e classificação
- O melhor modelo é selecionado através de busca em grade (GridSearchCV)
- Três algoritmos são testados: Naive Bayes, SVM e Random Forest
- O modelo é salvo em disco após o treinamento
- O modelo salvo é carregado para classificar novas perguntas