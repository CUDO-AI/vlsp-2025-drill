import numpy as np
import json
import os
from typing import List, Dict, Any
from .storage import FaissStorage
from core.logger import logger


class FaissIndexer:
    """
    FAISS Indexer class for integrating with existing indexing system
    """
    
    def __init__(self, dimension: int = 1024, m: int = 32, ef_construction: int = 200):
        """
        Initialize FAISS indexer
        
        Args:
            dimension: Dimension of embeddings
            m: Number of connections per layer in HNSW
            ef_construction: Construction parameter for HNSW
        """
        self.dimension = dimension
        self.m = m
        self.ef_construction = ef_construction
        self.storage = FaissStorage(dimension, m, ef_construction)
        
    def convert_chunks(self, corpus_path: str, embedding_path: str) -> tuple:
        """
        Convert corpus chunks and embeddings to format suitable for FAISS
        
        Args:
            corpus_path: Path to corpus JSON file
            embedding_path: Path to embeddings numpy file
            
        Returns:
            Tuple of (documents, embeddings)
        """
        with open(corpus_path) as f:
            corpus_chunks = json.load(f)
            
        corpus_embeddings = np.load(embedding_path)
        
        if len(corpus_chunks) != len(corpus_embeddings):
            raise ValueError("Number of chunks and embeddings must match")
            
        documents = []
        for chunk in corpus_chunks:
            if isinstance(chunk, str):
                title, content = "", chunk
            else:
                title = chunk.get("passage_title", "")
                content = chunk.get("passage_content", "")
                
            document = {
                "title": title,
                "content": content,
                "source": "corpus",
                "timestamp": ""
            }
            documents.append(document)
            
        return documents, corpus_embeddings
        
    def indexing(self, corpus_path: str, embedding_path: str, 
                index_dir: str = "faiss_storage", index_name: str = "faiss_index",
                overwrite: bool = False) -> str:
        """
        Index documents using FAISS HNSW
        
        Args:
            corpus_path: Path to corpus file
            embedding_path: Path to embeddings file
            index_dir: Directory to save index
            index_name: Name of the index
            overwrite: Whether to overwrite existing index
            
        Returns:
            Path to saved index
        """
        # Create index directory
        os.makedirs(index_dir, exist_ok=True)
        
        index_path = os.path.join(index_dir, f"{index_name}.faiss")
        documents_path = os.path.join(index_dir, f"{index_name}_documents.pkl")
        
        # Check if index exists
        if os.path.exists(index_path) and not overwrite:
            logger.info(f"Index {index_path} already exists. Use overwrite=True to recreate.")
            return index_path
            
        # Convert chunks and embeddings
        documents, embeddings = self.convert_chunks(corpus_path, embedding_path)
        
        logger.info(f"Converting {len(documents)} documents with {embeddings.shape[1]}-dimensional embeddings")
        
        # Add documents to FAISS index
        self.storage.add_documents(documents, embeddings)
        
        # Save index
        self.storage.save_index(index_path, documents_path)
        
        # Get statistics
        stats = self.storage.get_index_stats()
        logger.info(f"Indexed {stats['total_documents']} documents")
        
        return index_path
        
    def load_index(self, index_path: str, documents_path: str):
        """
        Load existing FAISS index
        
        Args:
            index_path: Path to FAISS index file
            documents_path: Path to documents file
        """
        self.storage.load_index(index_path, documents_path)
        logger.info(f"Loaded index from {index_path}")
        
    def search(self, query_embedding: np.ndarray, k: int = 10, 
               score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search in the loaded index
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of search results
        """
        return self.storage.search(query_embedding, k)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return self.storage.get_index_stats()
        
    def batch_indexing(self, corpus_paths: List[str], embedding_paths: List[str],
                      index_dir: str = "faiss_storage", index_name: str = "faiss_index",
                      overwrite: bool = False) -> str:
        """
        Batch index multiple corpus files
        
        Args:
            corpus_paths: List of corpus file paths
            embedding_paths: List of embedding file paths
            index_dir: Directory to save index
            index_name: Name of the index
            overwrite: Whether to overwrite existing index
            
        Returns:
            Path to saved index
        """
        if len(corpus_paths) != len(embedding_paths):
            raise ValueError("Number of corpus paths must match number of embedding paths")
            
        # Create index directory
        os.makedirs(index_dir, exist_ok=True)
        
        index_path = os.path.join(index_dir, f"{index_name}.faiss")
        documents_path = os.path.join(index_dir, f"{index_name}_documents.pkl")
        
        # Check if index exists
        if os.path.exists(index_path) and not overwrite:
            logger.info(f"Index {index_path} already exists. Use overwrite=True to recreate.")
            return index_path
            
        all_documents = []
        all_embeddings = []
        
        # Process each corpus file
        for corpus_path, embedding_path in zip(corpus_paths, embedding_paths):
            documents, embeddings = self.convert_chunks(corpus_path, embedding_path)
            all_documents.extend(documents)
            all_embeddings.append(embeddings)
            
        # Concatenate all embeddings
        all_embeddings = np.vstack(all_embeddings)
        
        logger.info(f"Batch indexing {len(all_documents)} documents with {all_embeddings.shape[1]}-dimensional embeddings")
        
        # Add documents to FAISS index
        self.storage.add_documents(all_documents, all_embeddings)
        
        # Save index
        self.storage.save_index(index_path, documents_path)
        
        # Get statistics
        stats = self.storage.get_index_stats()
        logger.info(f"Batch indexed {stats['total_documents']} documents")
        
        return index_path 