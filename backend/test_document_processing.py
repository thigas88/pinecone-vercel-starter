"""
Test suite for document processing and ingestion functionality.
"""

import requests
import json
import time
from pathlib import Path
import tempfile

API_URL = "http://localhost:8000"

def test_document_ingestion():
    """Test document ingestion with text content."""
    print("\n=== Testing Document Ingestion ===")
    
    # Sample document content
    document_content = """
    # Introduction to Machine Learning
    
    Machine learning is a subset of artificial intelligence (AI) that provides systems 
    the ability to automatically learn and improve from experience without being 
    explicitly programmed.
    
    ## Types of Machine Learning
    
    ### Supervised Learning
    Supervised learning is the machine learning task of learning a function that maps 
    an input to an output based on example input-output pairs.
    
    ### Unsupervised Learning
    Unsupervised learning is a type of machine learning that looks for previously 
    undetected patterns in a data set with no pre-existing labels.
    
    ### Reinforcement Learning
    Reinforcement learning is an area of machine learning concerned with how software 
    agents ought to take actions in an environment.
    
    ## Applications
    
    Machine learning has numerous applications including:
    - Natural Language Processing
    - Computer Vision
    - Recommendation Systems
    - Fraud Detection
    - Medical Diagnosis
    """
    
    # Test different splitting methods
    splitting_methods = ["character", "sentence", "semantic", "markdown"]
    
    for method in splitting_methods:
        print(f"\nTesting {method} splitting...")
        
        payload = {
            "content": document_content,
            "url": 'https://python.langchain.com/docs/integrations/vectorstores/timescalevector/',
            "title": f"ML Introduction - {method} split",
            "category": "technology",
            "tags": ["machine-learning", "ai", "tutorial"],
            "keywords": ["ML", "AI", "supervised", "unsupervised"],
            "splitting_method": method,
            "chunk_size": 100,
            "chunk_overlap": 20,
            "semantic_threshold": 0.5,
            "file_type": "markdown"
        }
        
        # Send ingestion request
        response = requests.post(f"{API_URL}/admin/ingest/document", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Document ingested: ID={result['document_id']}, Status={result['status']}")
            
            # Wait a bit for processing
            time.sleep(2)
            
            # Check chunks
            chunks_response = requests.get(
                f"{API_URL}/admin/documents/{result['document_id']}/chunks"
            )
            
            if chunks_response.status_code == 200:
                chunks = chunks_response.json()
                print(f"  Chunks created: {len(chunks)}")
                
                # Display first chunk
                if chunks:
                    print(f"  First chunk preview: {chunks[0]['chunk_text'][:100]}...")
        else:
            print(f"✗ Failed to ingest document: {response.status_code} - {response.text}")


def test_file_ingestion():
    """Test file ingestion with different file types."""
    print("\n=== Testing File Ingestion ===")
    
    # Create test files
    test_files = {
        "test.txt": "This is a test text file. It contains some sample content for testing the file ingestion system.",
        "test.md": "# Test Markdown\n\nThis is a **test** markdown file with *formatting*.\n\n## Section 1\nContent here.",
        "test.html": "<html><body><h1>Test HTML</h1><p>This is a test HTML file.</p></body></html>"
    }
    
    for filename, content in test_files.items():
        print(f"\nTesting {filename} ingestion...")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix=Path(filename).suffix, delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            # Upload file
            with open(temp_path, 'rb') as f:
                files = {'file': (filename, f, 'text/plain')}
                data = {
                    'title': f'Test {filename}',
                    'category': 'test',
                    'tags': ['test', 'file'],
                    'splitting_method': 'character',
                    'chunk_size': '100',
                    'chunk_overlap': '20'
                }
                
                response = requests.post(
                    f"{API_URL}/admin/ingest/file",
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✓ File ingested: ID={result['document_id']}")
                else:
                    print(f"✗ Failed to ingest file: {response.status_code} - {response.text}")
        
        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)


def test_search_functionality():
    """Test vector search functionality."""
    print("\n=== Testing Search Functionality ===")
    
    # First, ensure we have some documents
    test_queries = [
        "machine learning algorithms",
        "supervised learning examples",
        "AI applications",
        "neural networks"
    ]
    
    for query in test_queries:
        print(f"\nSearching for: '{query}'")
        
        response = requests.get(
            f"{API_URL}/search",
            params={
                "query": query,
                "k": 3,
                "score_threshold": 0.5
            }
        )
        
        if response.status_code == 200:
            results = response.json()
            print(f"✓ Found {results['count']} results")
            
            for i, result in enumerate(results['results'][:2]):
                print(f"\n  Result {i+1}:")
                print(f"  Score: {result['score']:.3f}")
                print(f"  Content: {result['content'][:100]}...")
                print(f"  Category: {result['metadata'].get('category', 'N/A')}")
        else:
            print(f"✗ Search failed: {response.status_code} - {response.text}")


def test_vector_store_stats():
    """Test vector store statistics endpoint."""
    print("\n=== Testing Vector Store Stats ===")
    
    response = requests.get(f"{API_URL}/admin/vector-store/stats")
    
    if response.status_code == 200:
        stats = response.json()
        print("✓ Vector Store Statistics:")
        print(f"  Total chunks: {stats.get('total_chunks', 0)}")
        print(f"  Total documents: {stats.get('total_documents', 0)}")
        print(f"  Table size: {stats.get('table_size', 'N/A')}")
        
        if stats.get('category_distribution'):
            print("\n  Category Distribution:")
            for cat in stats['category_distribution']:
                print(f"    {cat['category']}: {cat['count']} chunks")
    else:
        print(f"✗ Failed to get stats: {response.status_code} - {response.text}")


def test_error_handling():
    """Test error handling for invalid requests."""
    print("\n=== Testing Error Handling ===")
    
    # Test invalid splitting method
    print("\nTesting invalid splitting method...")
    payload = {
        "content": "Test content",
        "title": "Test",
        "splitting_method": "invalid_method"
    }
    
    response = requests.post(f"{API_URL}/admin/ingest/document", json=payload)
    if response.status_code == 422:
        print("✓ Correctly rejected invalid splitting method")
    else:
        print(f"✗ Unexpected response: {response.status_code}")
    
    # Test invalid chunk size
    print("\nTesting invalid chunk size...")
    payload = {
        "content": "Test content",
        "title": "Test",
        "chunk_size": 50  # Too small
    }
    
    response = requests.post(f"{API_URL}/admin/ingest/document", json=payload)
    if response.status_code == 422:
        print("✓ Correctly rejected invalid chunk size")
    else:
        print(f"✗ Unexpected response: {response.status_code}")


def test_document_reindexing():
    """Test document reindexing functionality."""
    print("\n=== Testing Document Reindexing ===")
    
    # First create a document
    payload = {
        "content": "This is a test document for reindexing.",
        "title": "Reindex Test",
        "category": "test",
        "splitting_method": "character",
        "chunk_size": 100
    }
    
    response = requests.post(f"{API_URL}/admin/ingest/document", json=payload)
    
    if response.status_code == 200:
        doc_id = response.json()['document_id']
        print(f"✓ Created document: ID={doc_id}")
        
        # Wait for initial processing
        time.sleep(2)
        
        # Trigger reindexing
        reindex_response = requests.post(f"{API_URL}/admin/documents/{doc_id}/reindex")
        
        if reindex_response.status_code == 200:
            print("✓ Reindexing triggered successfully")
        else:
            print(f"✗ Reindexing failed: {reindex_response.status_code}")
    else:
        print(f"✗ Failed to create document: {response.status_code}")


def run_all_tests():
    """Run all tests."""
    print("Starting Document Processing Tests...")
    print("=" * 50)
    
    # Check if API is running
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code != 200:
            print("✗ API is not responding. Please start the backend server.")
            return
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Please start the backend server.")
        return
    
    # Run tests
    test_document_ingestion()
    # test_file_ingestion()
    # test_search_functionality()
    # test_vector_store_stats()
    # test_error_handling()
     #test_document_reindexing()
    
    print("\n" + "=" * 50)
    print("Tests completed!")


if __name__ == "__main__":
    run_all_tests()