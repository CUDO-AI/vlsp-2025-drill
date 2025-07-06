from typing import List
import requests

from rerankers.base import BaseRanker, RankerError
    

class NVIDIARanker(BaseRanker):
    def __init__(self, base_url: str, api_key: str = ""):
        """
        Initialize NVIDIA Ranker client
        """
        super().__init__(base_url, api_key)
        self.headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }

    def rerank(self, query: str, passages: List[str], model_name: str = ""):
        if not model_name:
            raise RankerError("Please specify a model name.")
        payload = {
            "model": model_name,
            "query": {
                "text": query
            },
            "passages": [{"text": passage} for passage in passages],
            "truncate": "END"
        }
        try:
            session = requests.Session()
            response = session.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()['rankings']
        except Exception as e:
            raise RankerError(f"Failed to rerank: {str(e)}")