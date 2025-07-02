from typing import Dict


def weight_sum_score(sparse_result: Dict, dense_result: Dict, top_k: int = 100, weight_on_dense: float = 0.6):
    """Combined score between BM25 and Dense, this function was inspired by pyserini"""
    hybrid_result = {}
    # Find the min and max scores for sparse and dense results
    min_dense_score = min(dense_result.values()) if len(dense_result) > 1 else 0
    max_dense_score = max(dense_result.values()) if len(dense_result) > 1 else 1

    min_sparse_score = min(sparse_result.values()) if len(sparse_result) > 1 else 0
    max_sparse_score = max(sparse_result.values()) if len(sparse_result) > 1 else 1
    for psg in set(dense_result.keys()) | set(sparse_result.keys()):
        sparse_score = sparse_result.get(psg, min_sparse_score)
        dense_score = dense_result.get(psg, min_dense_score)

        # Normalize scores between 0 and 1
        sparse_score = (sparse_score - min_sparse_score) / (max_sparse_score - min_sparse_score)
        dense_score = (dense_score - min_dense_score) / (max_dense_score - min_dense_score)

        # Combine scores using the given weight
        score = weight_on_dense * dense_score + (1 - weight_on_dense) * sparse_score
        hybrid_result[psg] = (score, sparse_score, dense_score)

    # Sort and return the top-k results
    return dict(sorted(hybrid_result.items(), key=lambda x: x[1][0], reverse=True)[:top_k])
