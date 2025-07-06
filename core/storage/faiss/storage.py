import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Optional
from core.logger import logger


class FaissStorage:
    """
    FAISS Storage class for indexing and managing HNSW index
    """
    
    def __init__(self, dimension: int = 1024, m: int = 32, ef_construction: int = 200):
        """
        Initialize FAISS HNSW index
        
        Args:
            dimension: Dimension of vectors
            m: Number of connections per layer (default: 32)
            ef_construction: Size of the dynamic candidate list during construction (default: 200)
        """
        self.dimension = dimension
        self.m = m
        self.ef_construction = ef_construction
        self.index = None
        self.documents = []
        self.metadata = []
        
    def create_index(self):
        """Create HNSW index"""
        # Create HNSW index
        self.index = faiss.IndexHNSWFlat(self.dimension, self.m)
        
        # Set construction parameters
        self.index.hnsw.efConstruction = self.ef_construction
        
        # Set search parameters
        self.index.hnsw.efSearch = 128
        
        logger.info(f"Created HNSW index with dimension={self.dimension}, m={self.m}, ef_construction={self.ef_construction}")
        
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the index
        
        Args:
            documents: List of document dictionaries
            embeddings: numpy array of embeddings with shape (n_docs, dimension)
        """
        if self.index is None:
            self.create_index()
            
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
            
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        self.documents.extend(documents)
        
        # Create metadata for each document
        for i, doc in enumerate(documents):
            metadata = {
                'id': len(self.metadata) + i,
                'title': doc.get('title', ''),
                'content': doc.get('content', ''),
                'source': doc.get('source', ''),
                'timestamp': doc.get('timestamp', '')
            }
            self.metadata.append(metadata)
            
        logger.info(f"Added {len(documents)} documents to FAISS index")
        
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of dictionaries containing document info and similarity scores
        """
        if self.index is None:
            raise ValueError("Index not created. Call create_index() first.")
            
        # Ensure query embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:  # Valid index
                result = {
                    'id': idx,
                    'score': float(1.0 / (1.0 + distance)),  # Convert distance to similarity score
                    'distance': float(distance),
                    'document': self.documents[idx] if idx < len(self.documents) else None,
                    'metadata': self.metadata[idx] if idx < len(self.metadata) else None
                }
                results.append(result)
                
        return results
    
    def save_index(self, index_path: str, documents_path: str):
        """
        Save index and documents to disk
        
        Args:
            index_path: Path to save FAISS index
            documents_path: Path to save documents
        """
        if self.index is None:
            raise ValueError("No index to save")
            
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save documents and metadata
        with open(documents_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'dimension': self.dimension,
                'm': self.m,
                'ef_construction': self.ef_construction
            }, f)
            
        logger.info(f"Saved index to {index_path} and documents to {documents_path}")
        
    def load_index(self, index_path: str, documents_path: str):
        """
        Load index and documents from disk
        
        Args:
            index_path: Path to FAISS index
            documents_path: Path to documents file
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load documents and metadata
        with open(documents_path, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
            self.dimension = data['dimension']
            self.m = data['m']
            self.ef_construction = data['ef_construction']
            
        logger.info(f"Loaded index from {index_path} with {len(self.documents)} documents")
        
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index"""
        if self.index is None:
            return {'error': 'Index not created'}
            
        return {
            'total_documents': len(self.documents),
            'index_size': self.index.ntotal,
            'dimension': self.dimension,
            'm': self.m,
            'ef_construction': self.ef_construction,
            'is_trained': self.index.is_trained
        } 