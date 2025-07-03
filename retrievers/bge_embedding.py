from retrievers.base_embedding import BaseEmbedding
from FlagEmbedding import BGEM3FlagModel


class BGEEmbedding(BaseEmbedding):
    def __init__(self, model_name_or_path: str, base_url: str, model_name: str):
        super().__init__(base_url, model_name)
        self.encoder = BGEM3FlagModel(model_name_or_path, pooling_method="mean", normalize_embeddings=True)
    
    def embedd_query(self, query: str, max_length: int = 512):
        return self._embedd([query], max_length=max_length)[0]

    def embedd_queries(self, queries: list[str], batch_size: int = 32):
        return self._embedd(queries, batch_size=batch_size)
    
    def embedd_passage(self, passage: str, max_length: int = 512):
        return self._embedd([passage], max_length=max_length)[0]

    def embedd_passages(self, passages: list[str], batch_size: int = 32, max_length: int = 512):
        return self._embedd(passages, batch_size=batch_size, max_length=max_length)
    
    def _embedd(self, texts: list[str], batch_size: int = 32, max_length: int = 512):
        embeddings = self.encoder.encode(texts, batch_size=batch_size, return_dense=True, max_length=max_length)['dense_vecs']
        embeddings = embeddings.tolist()
        return embeddings