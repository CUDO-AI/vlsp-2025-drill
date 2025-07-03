from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url
        self.model_name = model_name
    
    @abstractmethod
    def embedd_query(self, query: str):
        raise NotImplementedError

    @abstractmethod
    def embedd_queries(self, queries: list[str], batch_size: int = 32):
        raise NotImplementedError
    
    @abstractmethod
    def embedd_passage(self, passage: str):
        raise NotImplementedError

    @abstractmethod
    def embedd_passages(self, passages: list[str], batch_size: int = 32):
        raise NotImplementedError
