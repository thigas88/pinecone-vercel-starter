"""
Vector store integration using TimescaleVector and LangChain.
Based on: https://python.langchain.com/docs/integrations/vectorstores/timescalevector/
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
import uuid

from langchain_community.vectorstores import TimescaleVector
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document as LangChainDocument
from langchain_community.vectorstores.pgvector import DistanceStrategy
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
import numpy as np

from config import logger


class VectorStoreManager:
    """Manages vector storage operations using TimescaleVector."""
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        embedding_model: str = "huggingface",
        collection_name: str = "document_chunks_vector"
    ):
        """
        Initialize the vector store manager.
        
        Args:
            connection_string: PostgreSQL connection string
            embedding_model: Type of embedding model to use ("google" or "huggingface")
            collection_name: Name of the collection/table for vectors
        """
        self.connection_string = connection_string or os.getenv(
            'DATABASE_URL', 
            'postgresql://chatrag:chatrag@localhost:5432/chatrag'
        )
        
        # Initialize embeddings
        if embedding_model == "google":
            os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_GENERATIVE_AI_API_KEY')
            model_embedding_name = os.getenv('MODEL_EMBEDDINGS_NAME', "text-embedding-004")
            self.embeddings = GoogleGenerativeAIEmbeddings(model=model_embedding_name)
        else:
            model_name = "rufimelo/bert-large-portuguese-cased-sts"  # melhor e mais pesado
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': False}
            # Use a smaller model for local embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
        
        self.collection_name = collection_name
        self.vector_store = None
        
        # Initialize the vector store
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize TimescaleVector store with proper schema."""
        try:
            # Create TimescaleVector instance
            self.vector_store = TimescaleVector(
                collection_name=self.collection_name,
                service_url=self.connection_string,
                embedding=self.embeddings,
                distance_strategy = DistanceStrategy.COSINE,
                pre_delete_collection=False,  # Don't delete existing data
                # Note: time partitioning is already configured in init_timescaledb.sql
            )
            
            # Ensure the table has the required columns
            self._ensure_table_schema()
            
            logger.info(f"Vector store initialized with collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def _ensure_table_schema(self):
        """Ensure the vector table has all required columns."""
        engine = create_engine(self.connection_string)
        
        with engine.connect() as conn:
            # Check if table exists
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = :table_name
                );
            """), {"table_name": self.collection_name})
            
            table_exists = result.scalar()
            
            if not table_exists:
                # Create table with TimescaleDB extensions
                # conn.execute(text(f"""
                #     CREATE TABLE IF NOT EXISTS {self.collection_name} (
                #         id UUID NOT NULL gen_random_uuid(),
                #         content TEXT NOT NULL,
                #         embedding vector(384),  -- Adjust dimension based on model
                #         metadata JSONB,
                #         document_id INTEGER,
                #         chunk_index INTEGER,
                #         created_at TIMESTAMPTZ DEFAULT NOW(),
                #         updated_at TIMESTAMPTZ DEFAULT NOW(),
                #         PRIMARY KEY (id, created_at)
                #     );
                    
                #     -- Create hypertable for time-series optimization
                #     SELECT create_hypertable(
                #         '{self.collection_name}', 
                #         'created_at',
                #         if_not_exists => TRUE
                #     );
                    
                #     -- Create indexes for better performance
                #     CREATE INDEX IF NOT EXISTS idx_{self.collection_name}_embedding 
                #     ON {self.collection_name} 
                #     USING ivfflat (embedding vector_cosine_ops)
                #     WITH (lists = 100);
                    
                #     CREATE INDEX IF NOT EXISTS idx_{self.collection_name}_document_id 
                #     ON {self.collection_name} (document_id);
                    
                #     CREATE INDEX IF NOT EXISTS idx_{self.collection_name}_metadata 
                #     ON {self.collection_name} 
                #     USING gin (metadata);
                # """))
                
                # conn.commit()
                # logger.info(f"Created vector table: {self.collection_name}")
                logger.info(f"Vector table: {self.collection_name} not exist")
    
    def add_documents(
        self,
        chunks: List[Dict[str, Any]],
        document_id: int,
        batch_size: int = 100
    ) -> List[str]:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of processed chunks with text and metadata
            document_id: ID of the parent document
            batch_size: Number of chunks to process at once
            
        Returns:
            List of chunk IDs
        """
        chunk_ids = []
        
        try:
            # Process chunks in batches
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                # Convert to LangChain documents
                documents = []
                metadatas = []
                
                for chunk in batch:
                    # Prepare metadata
                    metadata = chunk["metadata"].copy()
                    metadata["document_id"] = document_id
                    metadata["chunk_hash"] = chunk["hash"]
                    metadata["created_at"] = datetime.utcnow().isoformat()
                    
                    # Create LangChain document
                    doc = LangChainDocument(
                        page_content=chunk["text"],
                        metadata=metadata,
                        document_id=document_id,
                        chunck_id=chunk["index"]
                    )
                    documents.append(doc)
                    # metadatas.append(metadata)
                
                # Add to vector store
                ids = self.vector_store.add_documents(
                    documents=documents,
                    ids=[str(uuid.uuid4()) for _ in documents]
                )
                
                chunk_ids.extend(ids)
                
                logger.info(f"Added batch of {len(documents)} chunks to vector store")
            
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[LangChainDocument, float]]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Metadata filters
            score_threshold: Minimum similarity score
            
        Returns:
            List of (document, score) tuples
        """
        try:
            # Build filter for TimescaleVector
            # Ref https://python.langchain.com/docs/integrations/vectorstores/pgvector/

           
            filter_query = None
            if filter_dict:
                # Constrói um dicionário com todos os filtros
                # A chave é o campo nos metadados, o valor é a condição
                filters = {}
                for key, value in filter_dict.items():
                    # @todo filter by tags not functional
                    if isinstance(value, list):
                        filters[key] = {"$in": value}
                    else:
                        filters[key] = value
                
                # A estrutura final do filtro para a biblioteca PGVector
                filter_query = filters
            
            logger.info(f"Filter query: {filter_query}")

            # Perform search
            results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter_query
                )
            if score_threshold:                
                # Filter by score threshold
                # similarity_search_with_score retorna uma distância, onde 0 é mais similar.
                # 1 - score, converte a distância de cosseno em uma similaridade, onde 1 é mais similar.
                results = [(doc, score) for doc, score in results if 1 - score >= score_threshold]
                print(f"Filter by score threshold {score_threshold}")
                        
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            raise
    
    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        alpha: float = 0.5
    ) -> List[Tuple[LangChainDocument, float]]:
        """
        Perform hybrid search combining vector similarity and time-based filtering.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Metadata filters
            time_range: Tuple of (start_time, end_time) for temporal filtering
            alpha: Weight for combining vector and time scores (0-1)
            
        Returns:
            List of (document, score) tuples
        """
        try:
            # Add time range to filters if provided
            if time_range and filter_dict is None:
                filter_dict = {}
            
            if time_range:
                start_time, end_time = time_range
                # Ensure start_time and end_time are timezone-aware UTC for consistency
                # and pass datetime objects directly to filter_dict.
                # The underlying similarity_search method is assumed to handle
                # conversion to database-specific date formats if necessary.
                if start_time.tzinfo is None:
                    start_time = start_time.replace(tzinfo=timezone.utc)
                if end_time.tzinfo is None:
                    end_time = end_time.replace(tzinfo=timezone.utc)

                filter_dict["created_at"] = {
                    "$gte": start_time,
                    "$lte": end_time
                }
            
            # Perform vector search
            vector_results = self.similarity_search(
                query=query,
                k=k * 2,  # Get more results for re-ranking
                filter_dict=filter_dict
            )
            
            if not time_range or alpha == 1.0:
                # No time-based scoring needed
                return vector_results[:k]
            
            # Re-rank based on combined score
            reranked_results = []
            # Use timezone-aware UTC current time for consistent arithmetic
            current_time = datetime.now(timezone.utc) 
            
            for doc, vector_score in vector_results:
                # Calculate time-based score (more recent = higher score)
                doc_created_at_value = doc.metadata.get("created_at")
                
                # Robustly convert created_at from metadata to a datetime object
                if isinstance(doc_created_at_value, datetime):
                    created_at_dt = doc_created_at_value
                elif isinstance(doc_created_at_value, str):
                    try:
                        created_at_dt = datetime.fromisoformat(doc_created_at_value)
                    except ValueError:
                        # Fallback if string is not a valid ISO format
                        # Use current_time as a fallback for the date, ensuring it's timezone-aware
                        created_at_dt = current_time 
                else:
                    # Fallback for None or unexpected types in metadata
                    created_at_dt = current_time # Fallback to timezone-aware current_time

                # Ensure created_at_dt is timezone-aware UTC for consistent arithmetic
                if created_at_dt.tzinfo is None:
                    created_at_dt = created_at_dt.replace(tzinfo=timezone.utc)

                # Calculate time difference and score
                time_diff = (current_time - created_at_dt).total_seconds()
                max_time_diff = 30 * 24 * 3600  # 30 days in seconds (adjust as needed)
                time_score = max(0, 1 - (time_diff / max_time_diff))
                
                # Combine scores
                combined_score = alpha * vector_score + (1 - alpha) * time_score
                reranked_results.append((doc, combined_score))
            
            # Sort by combined score and return top k
            reranked_results.sort(key=lambda x: x[1], reverse=True)
            return reranked_results[:k]
            
        except Exception as e:
            # Using a generic logger.error as per original code, assuming it's defined.
            # In a real app, you might want to import logging and configure it.
            print(f"ERROR: Failed to perform hybrid search: {e}") # Using print for demonstration
            raise
    
    def delete_document_chunks(self, document_id: int) -> int:
        """
        Delete all chunks for a specific document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            Number of chunks deleted
        """
        try:
            engine = create_engine(self.connection_string)
            
            with engine.connect() as conn:
                result = conn.execute(text(f"""
                    DELETE FROM {self.collection_name}
                    WHERE metadata->>'document_id' = :doc_id
                """), {"doc_id": str(document_id)})
                
                conn.commit()
                deleted_count = result.rowcount
                
                logger.info(f"Deleted {deleted_count} chunks for document {document_id}")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to delete document chunks: {e}")
            raise
    
    def update_chunk_metadata(
        self,
        chunk_id: str,
        metadata_updates: Dict[str, Any]
    ) -> bool:
        """
        Update metadata for a specific chunk.
        
        Args:
            chunk_id: ID of the chunk
            metadata_updates: Dictionary of metadata fields to update
            
        Returns:
            True if successful
        """
        try:
            engine = create_engine(self.connection_string)
            
            with engine.connect() as conn:
                # Merge metadata updates
                result = conn.execute(text(f"""
                    UPDATE {self.collection_name}
                    SET metadata = metadata || :updates,
                        updated_at = NOW()
                    WHERE id = :chunk_id
                """), {
                    "chunk_id": chunk_id,
                    "updates": metadata_updates
                })
                
                conn.commit()
                
                return result.rowcount > 0
                
        except Exception as e:
            logger.error(f"Failed to update chunk metadata: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection."""
        try:
            engine = create_engine(self.connection_string)            
            stats = {}

            with engine.connect() as conn:
                # Get basic stats
                result = conn.execute(text(f"""
                    SELECT 
                        COUNT(*) as total_chunks,
                        COUNT(DISTINCT metadata->>'document_id') as total_documents,
                        MIN(created_at) as oldest_chunk,
                        MAX(created_at) as newest_chunk,
                        pg_size_pretty(pg_total_relation_size('{self.collection_name}')) as table_size
                    FROM {self.collection_name}
                """))
                data = result.fetchone()
                
                if data:
                    stats = dict(data._mapping)
                                
                # Get category distribution
                result = conn.execute(text(f"""
                    SELECT 
                        metadata->>'category' as category,
                        COUNT(*) as count
                    FROM {self.collection_name}
                    WHERE metadata->>'category' IS NOT NULL
                    GROUP BY metadata->>'category'
                    ORDER BY count DESC
                """))
                
                stats["category_distribution"] = [
                    {"category": row[0], "count": row[1]} 
                    for row in result.fetchall()
                ]
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            raise