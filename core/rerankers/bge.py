import sys
import os
from contextlib import contextmanager


@contextmanager
def suppress_tqdm():
    # Store original stderr
    original_stderr = sys.stderr
    
    # Create a null device that won't be closed
    null_device = open(os.devnull, 'w')
    
    try:
        # Redirect stderr to null device
        sys.stderr = null_device
        yield
    finally:
        # Restore original stderr
        sys.stderr = original_stderr
        # Close the null device
        null_device.close()

# Suppress tqdm output during import
with suppress_tqdm():
    from FlagEmbedding import FlagReranker

import string
import numpy as np

from core.rerankers.base import BaseRanker


class BGEReranker(BaseRanker):
    def __init__(self, model_name_or_path: str):
        # Suppress tqdm output during model initialization
        with suppress_tqdm():
            self.reranker = FlagReranker(model_name_or_path=model_name_or_path, normalize=True, devices="cuda", use_fp16=True)

    def rerank(self, question: str, passages: list[dict], batch_size: int = 16, max_length: int = 512) -> list[dict]:
        pairs = self.preprocess_ranking_input(question, passages)
        # Suppress tqdm output during scoring
        with suppress_tqdm():
            scores = self.reranker.compute_score(pairs, batch_size=batch_size, max_length=max_length)
        if scores is None:
            scores = [0.0] * len(passages)
        elif isinstance(scores, np.ndarray):
            scores = scores.tolist()
        for psg, score in zip(passages, scores):
            psg["score"] = score
        ranked_passages = sorted(passages, key=lambda x: x["score"], reverse=True)
        return ranked_passages
    
    @staticmethod
    def preprocess_ranking_input(question: str, passages: list) -> list:
        def clean_text(text: str) -> str:
            text = text.replace("\n", " ")
            text = text.replace("\t", " ")
            text = text.replace("\r", " ")
            return " ".join(text.split())
        
        def is_not_punctuation(char):
            return char not in string.punctuation

        def process_passage(passage: str | dict):
            content_str = ""
            if isinstance(passage, dict):
                title = passage.get("title", "")
                # content = clean_text(passage.get("content", ""))
                content = passage.get("content", "")
                if title and is_not_punctuation(title[-1]):
                    content_str = f"{title}. {content}"
                else:
                    content_str = f"{title} {content}"
            else:
                return passage
            return content_str.strip()

        processed_passages = [process_passage(psg) for psg in passages]
        pairs = [[question, psg] for psg in processed_passages]
        return pairs
