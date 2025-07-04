from abc import ABC, abstractmethod
from typing import Optional, List


class BaseEmbedding(ABC):
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url
        self.model_name = model_name
    
    @abstractmethod
    def embedd_query(self, query: str, dimension: Optional[int] = None):
        raise NotImplementedError

    @abstractmethod
    def embedd_queries(self, queries: List[str], batch_size: int = 32, dimension: Optional[int] = None):
        raise NotImplementedError
    
    @abstractmethod
    def embedd_passage(self, passage: str, dimension: Optional[int] = None):
        raise NotImplementedError

    @abstractmethod
    def embedd_passages(self, passages: List[str], batch_size: int = 32, dimension: Optional[int] = None):
        raise NotImplementedError
