# Document Processing and Vector Storage System

This document describes the comprehensive document processing system that has been implemented for the ChatRAG backend. The system supports multiple text splitting strategies, metadata enrichment, and vector storage using TimescaleVector.

## Overview

The document processing system accepts complete documents and automatically:
1. Splits them into chunks using configurable strategies
2. Enriches chunks with metadata
3. Stores chunks in both relational and vector databases
4. Enables semantic search across documents

## Architecture

### Core Components

1. **Document Processor** (`document_processor.py`)
   - Handles document loading and text extraction
   - Implements multiple splitting strategies
   - Extracts keywords and enriches metadata

2. **Vector Store Manager** (`vector_store.py`)
   - Manages TimescaleVector integration
   - Handles vector embeddings and similarity search
   - Provides hybrid search capabilities

3. **API Endpoints** (`app.py`)
   - Document ingestion endpoints
   - Search functionality
   - Statistics and management

## Text Splitting Strategies

### 1. Character Text Splitter
- Fixed-length segmentation based on character count
- Configurable chunk size and overlap
- Best for: General text without specific structure

```python
{
    "splitting_method": "character",
    "chunk_size": 1000,
    "chunk_overlap": 200
}
```

### 2. Sentence Splitter
- Splits at sentence boundaries
- Maintains semantic coherence
- Best for: Natural language documents

```python
{
    "splitting_method": "sentence",
    "chunk_size": 1000,
    "chunk_overlap": 200
}
```

### 3. Semantic Splitter
- Context-aware chunking based on semantic similarity
- Uses sentence embeddings to group related content
- Best for: Technical documents, research papers

```python
{
    "splitting_method": "semantic",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "semantic_threshold": 0.5
}
```

### 4. Markdown Splitter
- Structure-preserving splitting for markdown documents
- Respects headers and formatting
- Best for: Documentation, markdown files

```python
{
    "splitting_method": "markdown",
    "chunk_size": 1000,
    "chunk_overlap": 200
}
```

## Metadata Schema

Each chunk is enriched with comprehensive metadata:

```json
{
    "category": "technology",           // Document classification
    "keywords": ["AI", "ML"],          // Extracted key terms
    "tags": ["tutorial", "guide"],     // User-defined labels
    "title": "Document Title",         // Document or section title
    "url": "https://source.com",       // Source location
    "url_ref": "https://...#chunk-0",  // Specific chunk reference
    "type": "pdf",                     // File format
    "chunk_index": 0,                  // Position in document
    "total_chunks": 10,                // Total chunks in document
    "source_hash": "abc123...",        // Document hash
    "created_at": "2024-01-01T00:00:00Z"
}
```

## API Endpoints

### 1. Document Ingestion

**POST** `/ingest/document`

Ingest a complete document with text content:

```json
{
    "content": "Document text content...",
    "title": "My Document",
    "category": "technology",
    "tags": ["AI", "tutorial"],
    "keywords": ["machine learning", "neural networks"],
    "splitting_method": "semantic",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "semantic_threshold": 0.5,
    "file_type": "text"
}
```

### 2. File Ingestion

**POST** `/ingest/file`

Upload and process a file (PDF, DOCX, TXT, HTML, MD):

```bash
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@document.pdf" \
  -F "title=My Document" \
  -F "category=technology" \
  -F "tags=AI,tutorial" \
  -F "splitting_method=character" \
  -F "chunk_size=1000" \
  -F "chunk_overlap=200"
```

### 3. Search Documents

**GET** `/search`

Search for documents using vector similarity:

```bash
curl "http://localhost:8000/search?query=machine%20learning&k=5&category=technology&score_threshold=0.7"
```

Response:
```json
{
    "query": "machine learning",
    "results": [
        {
            "content": "Chunk text...",
            "metadata": {...},
            "score": 0.85
        }
    ],
    "count": 5
}
```

### 4. Vector Store Statistics

**GET** `/vector-store/stats`

Get statistics about the vector store:

```json
{
    "total_chunks": 1500,
    "total_documents": 50,
    "oldest_chunk": "2024-01-01T00:00:00Z",
    "newest_chunk": "2024-01-15T00:00:00Z",
    "table_size": "125 MB",
    "category_distribution": [
        {"category": "technology", "count": 800},
        {"category": "science", "count": 700}
    ]
}
```

## Database Schema

### TimescaleDB Tables

1. **documents** - Main document records
2. **document_chunks** - Text chunks with references
3. **document_chunks_vector** - Vector embeddings for similarity search
4. **ingest_history** - Processing history and status

### Vector Storage

- Uses pgvector extension for efficient vector operations
- Supports up to 384-dimensional embeddings (all-MiniLM-L6-v2)
- IVFFlat index for fast similarity search
- Time-series optimization with TimescaleDB hypertables

## Configuration

### Environment Variables

```bash
DATABASE_URL=postgresql://user:pass@localhost:5432/chatrag
OPENAI_API_KEY=your-api-key  # Optional, for OpenAI embeddings
```

### Embedding Models

The system supports two embedding models:

1. **OpenAI Embeddings** (requires API key)
   - Higher quality embeddings
   - Dimension: 1536

2. **HuggingFace Embeddings** (local, default)
   - Model: sentence-transformers/all-MiniLM-L6-v2
   - Dimension: 384
   - No API key required

## Error Handling

The system includes comprehensive error handling:

- Validation of splitting parameters
- File type verification
- Chunk size limits (100-10000 characters)
- Async processing with status tracking
- Detailed error messages in ingest history

## Performance Considerations

1. **Batch Processing**: Chunks are processed in batches of 100
2. **Async Processing**: Document processing happens in background
3. **Time Partitioning**: Vector table partitioned by month
4. **Indexing**: Optimized indexes for search performance

## Testing

Run the comprehensive test suite:

```bash
python backend/test_document_processing.py
```

Tests include:
- Document ingestion with all splitting methods
- File upload processing
- Search functionality
- Error handling
- Statistics retrieval

## Example Usage

### Python Client Example

```python
import requests

# Ingest a document
response = requests.post(
    "http://localhost:8000/ingest/document",
    json={
        "content": "Your document content here...",
        "title": "AI Tutorial",
        "category": "technology",
        "tags": ["AI", "tutorial"],
        "splitting_method": "semantic",
        "chunk_size": 1000,
        "chunk_overlap": 200
    }
)

doc_id = response.json()["document_id"]

# Search for content
results = requests.get(
    "http://localhost:8000/search",
    params={
        "query": "machine learning algorithms",
        "k": 5,
        "category": "technology"
    }
).json()

for result in results["results"]:
    print(f"Score: {result['score']:.2f}")
    print(f"Content: {result['content'][:100]}...")
```

## Future Enhancements

1. **Additional Splitters**
   - Code-aware splitter for source code
   - Table-aware splitter for structured data
   - Language-specific splitters

2. **Advanced Features**
   - Multi-modal embeddings (text + images)
   - Cross-lingual search
   - Real-time document updates
   - Incremental indexing

3. **Performance**
   - GPU acceleration for embeddings
   - Distributed processing
   - Caching layer

## Troubleshooting

### Common Issues

1. **Vector dimension mismatch**
   - Ensure the embedding model matches the vector column dimension
   - Default is 384 for all-MiniLM-L6-v2

2. **Memory issues with large documents**
   - Reduce chunk_size
   - Process documents in smaller batches

3. **Slow search performance**
   - Ensure vector indexes are created
   - Consider adjusting IVFFlat lists parameter
   - Use time-based filtering to reduce search space

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License

This system is part of the ChatRAG project and follows the same license terms.