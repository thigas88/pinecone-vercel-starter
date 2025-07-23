import requests

API_URL = "http://localhost:8000"

# 1. Criar um documento
doc_payload = {
    "url": "https://exemplo.com/doc",
    "title": "Exemplo de Documento",
    "tags": "exemplo, teste",
    "category": "ecampus",
    "splitting_method": "markdown",
    "chunk_size": 256,
    "overlap": 1,
    "file_name": "exemplo.pdf"
}
resp = requests.post(f"{API_URL}/admin/documents", json=doc_payload)
resp.raise_for_status()
doc = resp.json()
document_id = doc["id"]
print("Documento criado:", doc)

# 2. Enviar chunks para o documento
chunks_payload = {
    "ingest_history_id": None,  # ou um ID válido se desejar
    "chunks": [
        {
            "chunk_index": 0,
            "chunk_text": "Primeiro trecho do documento.",
            "hash": "hash_do_chunk_0"
        },
        {
            "chunk_index": 1,
            "chunk_text": "Segundo trecho do documento.",
            "hash": "hash_do_chunk_1"
        }
    ]
}
resp = requests.post(f"{API_URL}/admin/documents/{document_id}/chunks", json=chunks_payload)
resp.raise_for_status()
print("Chunks enviados:", resp.json())

# 3. Consultar os chunks do documento
resp = requests.get(f"{API_URL}/admin/documents/{document_id}/chunks")
resp.raise_for_status()
chunks = resp.json()
print("Chunks do documento:")
for chunk in chunks:
    print(f"  [{chunk['chunk_index']}] {chunk['chunk_text'][:60]}...")

# 4. Consultar histórico de ingestão (opcional)
resp = requests.get(f"{API_URL}/admin/documents/{document_id}/history")
resp.raise_for_status()
history = resp.json()
print("Histórico de ingestão:", history)