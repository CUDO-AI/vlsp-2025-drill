from core.encoders.base import BaseEncoder
from FlagEmbedding import BGEM3FlagModel
import numpy as np


class BGEEncoder(BaseEncoder):
    def __init__(self, model_name_or_path: str, base_url: str = "", model_name: str = "",
                 prefix_query: str = "query", prefix_passage: str = "passage"):
        super().__init__(base_url, model_name)
        self.encoder = BGEM3FlagModel(model_name_or_path, pooling_method="mean", normalize_embeddings=True)
        self.prefix_query = prefix_query
        self.prefix_passage = prefix_passage
    
    def encode_query(
        self,
        query: str,
        max_length: int = 512,
        **kwargs
    ):
        return self._encode([query], max_length=max_length, prompt_name=self.prefix_query)[0]

    def encode_queries(
        self,
        queries: list[str],
        batch_size: int = 32,
        **kwargs
    ):
        return self._encode(queries, batch_size=batch_size, prompt_name=self.prefix_query)
    
    def encode_passage(
        self, 
        passage: str, 
        max_length: int = 512, 
        **kwargs
    ):
        return self._encode([passage], max_length=max_length, prompt_name=self.prefix_passage)[0]

    def encode_passages(
        self,
        passages: list[str],
        batch_size: int = 32,
        max_length: int = 512,
        **kwargs
    ):
        return self._encode(passages, batch_size=batch_size, max_length=max_length, prompt_name=self.prefix_passage)
    
    def _encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        max_length: int = 512,
        prompt_name: str = "",
        **kwargs
    ):
        if prompt_name.strip():
            texts = [f"{prompt_name}: {text}" for text in texts]
        embeddings = self.encoder.encode(texts, batch_size=batch_size, return_dense=True, max_length=max_length)['dense_vecs']
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        return embeddings