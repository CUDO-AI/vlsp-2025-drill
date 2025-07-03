from retrievers.base_embedding import BaseEmbedding

from typing import List


class GenerateEmbeddings:
    def __init__(self, embedder: BaseEmbedding):
        self.embedder = embedder

    def generate_embeddings(self, texts: List[str], batch_size: int = 32):
        return self.embedder.embedd_passages(texts, batch_size)