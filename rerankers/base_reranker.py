from typing import List
from abc import ABC, abstractmethod


class RankerError(Exception):
    """Custom exception for ranker errors"""
    pass


class BaseRanker(ABC):
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
    
    @abstractmethod
    def rerank(self, query: str, passages: List[str]):
        raise NotImplementedError
