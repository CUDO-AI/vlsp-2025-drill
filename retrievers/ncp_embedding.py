from retrievers.base_embedding import BaseEmbedding

from typing import List, Optional
import json
import requests
from tqdm import tqdm


class EmbeddingError(Exception):
    """Custom exception for embedding errors"""
    pass


class NCPEmbedding(BaseEmbedding):
    def __init__(self, base_url: str, api_key: str, model_name: str = ""):
        """
        Initialize NCP Embedding client
        """
        super().__init__(base_url, model_name)
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        self.base_url = f"{base_url.rstrip('/')}/embeddings"

    def embedd_query(self, query: str, dimension: Optional[int] = None):
        """Generate embedding for a single query"""
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
            
        try:
            query_embedding = self._embedd([query], "query")
            return query_embedding[0]['embedding']
        except Exception as e:
            raise EmbeddingError(f"Failed to embed query: {str(e)}")

    def embedd_queries(self, queries: List[str], batch_size: int = 32):
        """Generate embeddings for multiple queries"""
        if not queries:
            return []
        try:
            batch_inputs = [queries[i:i+batch_size] for i in range(0, len(queries), batch_size)]
            embeddings = []
            for batch in tqdm(batch_inputs, desc="Embedding queries"):
                batch_embeddings = self._embedd(batch, "query")
                embeddings.extend([emb['embedding'] for emb in batch_embeddings])
                
            return embeddings
        
        except Exception as e:
            raise EmbeddingError(f"Failed to embed queries: {str(e)}")

    def embedd_passage(self, passage: str, dimensions: Optional[int] = None):
        """Generate embedding for a single passage"""
        if not passage or not isinstance(passage, str):
            raise ValueError("Passage must be a non-empty string")
            
        try:
            passage_embedding = self._embedd([passage], "passage")
            return passage_embedding[0]['embedding']
        except Exception as e:
            raise EmbeddingError(f"Failed to embed passage: {str(e)}")

    def embedd_passages(self, passages: List[str], batch_size: int = 32):
        """Generate embeddings for multiple passages"""
        if not passages:
            return []    
        try:
            batch_inputs = [passages[i:i+batch_size] 
                          for i in range(0, len(passages), batch_size)]
            embeddings = []
            
            for batch in tqdm(batch_inputs, desc="Embedding passages"):
                batch_embeddings = self._embedd(batch, "passage")
                embeddings.extend([emb['embedding'] for emb in batch_embeddings])
                
            return embeddings
        
        except Exception as e:
            raise EmbeddingError(f"Failed to embed passages: {str(e)}")
    
    def _embedd(self, texts: List[str], input_type: str):
        """Internal method to call the embedding API"""
        texts = [f"{input_type}: {text}" for text in texts]
        payload = json.dumps({
            "input": texts,
            "model": self.model_name,
        })
        try:
            response = requests.post(
                url=self.base_url,
                headers=self.headers,
                data=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()["data"]
        
        except requests.RequestException as e:
            raise EmbeddingError(f"API request failed: {str(e)}")