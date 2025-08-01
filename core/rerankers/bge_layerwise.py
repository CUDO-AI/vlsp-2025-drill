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

with suppress_tqdm():
    from FlagEmbedding import LayerWiseFlagLLMReranker


import numpy as np

from core.rerankers.bge import BGEReranker


class BGELayerWiseReranker(BGEReranker):
    def __init__(self, model_name_or_path: str):
        with suppress_tqdm():
            self.reranker = LayerWiseFlagLLMReranker(model_name_or_path=model_name_or_path, use_fp16=True, normalize=True, devices="cuda")

    def rerank(self, question: str, passages: list[dict], batch_size: int = 16, max_length: int = 512) -> list[dict]:
        pairs = self.preprocess_ranking_input(question, passages)
        # Suppress tqdm output during scoring
        with suppress_tqdm():
            scores = self.reranker.compute_score(pairs, batch_size=batch_size, max_length=max_length, cutoff_layers=[28])
        if scores is None:
            scores = [0.0] * len(passages)
        elif isinstance(scores, np.ndarray):
            scores = scores.tolist()
        for psg, score in zip(passages, scores):
            psg["score"] = score
        ranked_passages = sorted(passages, key=lambda x: x["score"], reverse=True)
        return ranked_passages
