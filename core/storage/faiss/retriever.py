import numpy as np
from typing import List, Dict, Any, Optional
from .storage import FaissStorage
from core.logger import logger


class FaissRetriever:
    """
    FAISS Retriever class for document retrieval using HNSW index
    """
    
    def __init__(self, faiss_storage: FaissStorage):
        """
        Initialize FAISS retriever
        
        Args:
            faiss_storage: FaissStorage instance
        """
        self.storage = faiss_storage
        
    def retrieve(self, query_embedding: np.ndarray, k: int = 10, 
                 score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve documents using FAISS HNSW index
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of retrieved documents with scores
        """
        # Search in FAISS index
        results = self.storage.search(query_embedding, k)
        
        # Filter by score threshold
        filtered_results = [
            result for result in results 
            if result['score'] >= score_threshold
        ]
        
        logger.info(f"Retrieved {len(filtered_results)} documents (threshold: {score_threshold})")
        
        return filtered_results
    
    def batch_retrieve(self, query_embeddings: np.ndarray, k: int = 10,
                      score_threshold: float = 0.0) -> List[List[Dict[str, Any]]]:
        """
        Batch retrieve documents for multiple queries
        
        Args:
            query_embeddings: Batch of query embedding vectors
            k: Number of results per query
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of results for each query
        """
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
            
        batch_results = []
        for i, query_embedding in enumerate(query_embeddings):
            results = self.retrieve(query_embedding, k, score_threshold)
            batch_results.append(results)
            
        logger.info(f"Batch retrieved for {len(query_embeddings)} queries")
        
        return batch_results
    
    def get_top_k_documents(self, query_embedding: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """
        Get top-k documents without score filtering
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of top-k documents
        """
        results = self.storage.search(query_embedding, k)
        
        # Extract just the document content
        documents = []
        for result in results:
            doc_info = {
                'id': result['id'],
                'score': result['score'],
                'title': result['metadata']['title'] if result['metadata'] else '',
                'content': result['metadata']['content'] if result['metadata'] else '',
                'source': result['metadata']['source'] if result['metadata'] else ''
            }
            documents.append(doc_info)
            
        return documents
    
    def search_by_content(self, query_embedding: np.ndarray, k: int = 10,
                         content_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search with optional content filtering
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            content_filter: Optional content filter string
            
        Returns:
            List of filtered documents
        """
        results = self.storage.search(query_embedding, k * 2)  # Get more results for filtering
        
        if content_filter:
            # Simple content filtering
            filtered_results = []
            for result in results:
                content = result['metadata']['content'].lower() if result['metadata'] else ''
                if content_filter.lower() in content:
                    filtered_results.append(result)
                    if len(filtered_results) >= k:
                        break
            results = filtered_results[:k]
        
        return results[:k]
    
    def get_similarity_scores(self, query_embedding: np.ndarray, 
                            document_ids: List[int]) -> List[float]:
        """
        Get similarity scores for specific document IDs
        
        Args:
            query_embedding: Query embedding vector
            document_ids: List of document IDs to check
            
        Returns:
            List of similarity scores
        """
        if self.storage.index is None:
            raise ValueError("Index not loaded")
            
        valid_ids = []
        
        for doc_id in document_ids:
            if doc_id < len(self.storage.documents):
                # We need to get the embedding from the original data
                # This is a simplified version - in practice you'd store embeddings separately
                valid_ids.append(doc_id)
                
        if not valid_ids:
            return []
            
        # For this example, we'll do a full search and filter by IDs
        results = self.storage.search(query_embedding, len(self.storage.documents))
        
        # Create a mapping of doc_id to score
        id_to_score = {result['id']: result['score'] for result in results}
        
        # Return scores for requested IDs
        scores = []
        for doc_id in document_ids:
            scores.append(id_to_score.get(doc_id, 0.0))
            
        return scores 