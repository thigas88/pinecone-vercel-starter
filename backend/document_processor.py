"""
Document processing and text splitting module.
Implements multiple splitting strategies using LangChain.
"""

import hashlib
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import re

from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    SentenceTransformersTokenTextSplitter
)
from langchain.schema import Document as LangChainDocument
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader
)
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


@dataclass
class ChunkMetadata:
    """Metadata schema for document chunks."""
    category: Optional[str] = None
    keywords: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    title: Optional[str] = None
    url: Optional[str] = None
    url_ref: Optional[str] = None  # Specific chunk reference
    type: Optional[str] = None  # pdf, doc, web, markdown
    chunk_index: int = 0
    total_chunks: int = 0
    source_hash: Optional[str] = None


@dataclass
class SplittingConfig:
    """Configuration for document splitting."""
    method: str = "character"  # character, sentence, semantic, markdown
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separator: Optional[str] = None
    semantic_threshold: float = 0.5  # For semantic splitting


class DocumentSplitter(ABC):
    """Abstract base class for document splitters."""
    
    @abstractmethod
    def split(self, text: str, config: SplittingConfig) -> List[str]:
        """Split text into chunks based on the strategy."""
        pass


class CharacterSplitter(DocumentSplitter):
    """Fixed-length character-based text splitter."""
    
    def split(self, text: str, config: SplittingConfig) -> List[str]:
        splitter = CharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separator=config.separator or "\n\n",
            length_function=len
        )
        return splitter.split_text(text)


class SentenceSplitter(DocumentSplitter):
    """Sentence-boundary aware text splitter."""
    
    def split(self, text: str, config: SplittingConfig) -> List[str]:
        # Use NLTK to split into sentences first
        sentences = nltk.sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > config.chunk_size and current_chunk:
                # Join current chunk and add to chunks
                chunks.append(" ".join(current_chunk))
                
                # Start new chunk with overlap
                if config.chunk_overlap > 0:
                    # Calculate how many sentences to keep for overlap
                    overlap_text = " ".join(current_chunk)
                    if len(overlap_text) > config.chunk_overlap:
                        # Keep only the last part for overlap
                        overlap_sentences = []
                        overlap_length = 0
                        for sent in reversed(current_chunk):
                            if overlap_length + len(sent) <= config.chunk_overlap:
                                overlap_sentences.insert(0, sent)
                                overlap_length += len(sent)
                            else:
                                break
                        current_chunk = overlap_sentences
                        current_length = overlap_length
                    else:
                        current_chunk = []
                        current_length = 0
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks


class SemanticSplitter(DocumentSplitter):
    """Context-aware chunking based on semantic similarity."""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def split(self, text: str, config: SplittingConfig) -> List[str]:
        # First split into sentences
        sentences = nltk.sent_tokenize(text)
        
        if not sentences:
            return []
        
        # Get embeddings for all sentences
        embeddings = self.model.encode(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0].reshape(1, -1)
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_embedding = embeddings[i].reshape(1, -1)
            
            # Calculate similarity with current chunk
            chunk_embedding = np.mean(
                self.model.encode(current_chunk), 
                axis=0
            ).reshape(1, -1)
            
            similarity = cosine_similarity(sentence_embedding, chunk_embedding)[0][0]
            
            # Check if we should start a new chunk
            current_length = sum(len(s) for s in current_chunk)
            
            if (similarity < config.semantic_threshold and current_length > config.chunk_size // 2) or \
               (current_length + len(sentence) > config.chunk_size):
                # Save current chunk
                chunks.append(" ".join(current_chunk))
                
                # Start new chunk with overlap if needed
                if config.chunk_overlap > 0 and similarity >= config.semantic_threshold:
                    # Keep semantically similar sentences for overlap
                    overlap_sentences = []
                    overlap_length = 0
                    for sent in reversed(current_chunk):
                        if overlap_length + len(sent) <= config.chunk_overlap:
                            overlap_sentences.insert(0, sent)
                            overlap_length += len(sent)
                        else:
                            break
                    current_chunk = overlap_sentences + [sentence]
                else:
                    current_chunk = [sentence]
            else:
                current_chunk.append(sentence)
        
        # Add the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks


class MarkdownSplitter(DocumentSplitter):
    """Structure-preserving markdown document splitter."""
    
    def split(self, text: str, config: SplittingConfig) -> List[str]:
        # Use LangChain's MarkdownTextSplitter
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        
        splitter = MarkdownTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        return splitter.split_text(text)


class DocumentProcessor:
    """Main document processor that handles various file formats and splitting strategies."""
    
    def __init__(self):
        self.splitters = {
            "character": CharacterSplitter(),
            "sentence": SentenceSplitter(),
            "semantic": SemanticSplitter(),
            "markdown": MarkdownSplitter()
        }
    
    def load_document(self, file_path: str, file_type: str) -> str:
        """Load document content based on file type."""
        loaders = {
            "pdf": PyPDFLoader,
            "docx": Docx2txtLoader,
            "txt": TextLoader,
            "html": UnstructuredHTMLLoader,
            "md": UnstructuredMarkdownLoader,
            "markdown": UnstructuredMarkdownLoader
        }
        
        loader_class = loaders.get(file_type.lower())
        if not loader_class:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        loader = loader_class(file_path)
        documents = loader.load()
        
        # Combine all pages/sections into one text
        return "\n\n".join([doc.page_content for doc in documents])
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text using TF-IDF."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Simple keyword extraction
        vectorizer = TfidfVectorizer(
            max_features=max_keywords,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            return feature_names.tolist()
        except:
            # Fallback to simple word frequency
            words = re.findall(r'\b\w+\b', text.lower())
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, _ in sorted_words[:max_keywords]]
    
    def process_document(
        self,
        content: str,
        config: SplittingConfig,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process a document and return chunks with metadata."""
        
        # Select the appropriate splitter
        splitter = self.splitters.get(config.method)
        if not splitter:
            raise ValueError(f"Unknown splitting method: {config.method}")
        
        # Split the document
        text_chunks = splitter.split(content, config)
        
        # Generate document hash
        doc_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Extract keywords from the entire document
        doc_keywords = self.extract_keywords(content)
        
        # Process each chunk
        processed_chunks = []
        total_chunks = len(text_chunks)
        
        for idx, chunk_text in enumerate(text_chunks):
            # Generate chunk hash
            chunk_hash = hashlib.sha256(chunk_text.encode()).hexdigest()
            
            # Extract chunk-specific keywords
            chunk_keywords = self.extract_keywords(chunk_text, max_keywords=5)
            
            # Combine metadata
            chunk_metadata = ChunkMetadata(
                category=metadata.get("category"),
                keywords=list(set(doc_keywords + chunk_keywords)),
                tags=metadata.get("tags", []),
                title=metadata.get("title"),
                url=metadata.get("url"),
                url_ref=f"{metadata.get('url', '')}#chunk-{idx}",
                type=metadata.get("type", "text"),
                chunk_index=idx,
                total_chunks=total_chunks,
                source_hash=doc_hash
            )
            
            processed_chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata.__dict__,
                "hash": chunk_hash,
                "index": idx
            })
        
        return processed_chunks
    
    def process_file(
        self,
        file_path: str,
        file_type: str,
        config: SplittingConfig,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process a file and return chunks with metadata."""
        
        # Load document content
        content = self.load_document(file_path, file_type)
        
        # Add file type to metadata
        metadata["type"] = file_type
        
        # Process the document
        return self.process_document(content, config, metadata)