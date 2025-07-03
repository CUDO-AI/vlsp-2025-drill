from typing import List
import requests

from rerankers.base_reranker import BaseRanker, RankerError


class TEIRanker(BaseRanker):
    def __init__(self, base_url: str, api_key: str = ""):
        super().__init__(base_url, api_key)
        self.headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }

    def rerank(self, query: str, passages: List[str]):
        payload = {
            "query": query,
            "raw_scores": False,
            "return_text": False,
            "texts": passages,
            "truncate": False,
            "truncation_direction": "Right"
        }
        try:
            session = requests.Session()
            response = session.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RankerError(f"Failed to rerank: {str(e)}")
