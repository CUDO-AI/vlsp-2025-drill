from encoders.base import BaseEncoder
from FlagEmbedding import BGEM3FlagModel


class BGEEncoder(BaseEncoder):
    def __init__(self, model_name_or_path: str, base_url: str, model_name: str):
        super().__init__(base_url, model_name)
        self.encoder = BGEM3FlagModel(model_name_or_path, pooling_method="mean", normalize_embeddings=True)
    
    def encode_query(self, query: str, max_length: int = 512):
        return self._encode([query], max_length=max_length)[0]

    def encode_queries(self, queries: list[str], batch_size: int = 32):
        return self._encode(queries, batch_size=batch_size)
    
    def encode_passage(self, passage: str, max_length: int = 512):
        return self._encode([passage], max_length=max_length)[0]

    def encode_passages(self, passages: list[str], batch_size: int = 32, max_length: int = 512):
        return self._encode(passages, batch_size=batch_size, max_length=max_length)
    
    def _encode(self, texts: list[str], batch_size: int = 32, max_length: int = 512):
        embeddings = self.encoder.encode(texts, batch_size=batch_size, return_dense=True, max_length=max_length)['dense_vecs']
        embeddings = embeddings.tolist()
        return embeddings