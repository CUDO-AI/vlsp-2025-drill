from core.storage.elasticsearch.storage import ElasticsearchStore
from core.logger import logger
from core.encoders.base import BaseEncoder
from core.datasets.data_loader import load_test_data
from core.evaluation.eval_metrics import calculate_metrics_at_k

from tqdm.auto import tqdm
import time

import json


SLEEP_TIME = 20


def search(es_store: ElasticsearchStore, embedder: BaseEncoder, 
           eval_method: str, index: str, question: str, dimension: int):
    if eval_method == "dense":
        question_embedding = embedder.encode_query(question, dimension)
        hits = es_store.dense_search(index=index, query_embedding=question_embedding, top_k=100)["data"]
    elif eval_method == "sparse":
        hits = es_store.sparse_search(index=index, query=question, top_k=100)["data"]
    elif eval_method == "hybrid":
        question_embedding = embedder.encode_query(question, dimension)
        hits = es_store.hybrid_search(index=index, query=question, query_embedding=question_embedding, top_k=100)["data"]
    else:
        raise ValueError("Invalid evaluation method")
    return hits


def evaluate(es_store: ElasticsearchStore, embedder: BaseEncoder, eval_method: str, 
             data_path: str, index: str, log_path: str, save_path: str, dimension: int):
    logger.info('Loading questions...!')
    questions, answer_ids = load_test_data(data_path)
    logger.info('Example for question: \n' + "\n".join(questions[:5]))
    assert len(answer_ids) == len(questions)
    logger.info('Starting evaluate...!')
    report_top_k = [1, 3, 5, 10, 20, 50]
    start = time.time()
    result = []
    predictions = []
    for relevant, question in tqdm(zip(answer_ids, questions), total=len(questions)):
        hits = search(es_store, embedder, eval_method, index, question, dimension)
        result.append({"question": question, "retrieval_hits": hits, "relevants": relevant})
        hit_indices = [str(hit) for hit in hits]
        predictions.append(hit_indices)
    calculate_metrics_at_k(predictions, answer_ids, report_top_k, log_path)
    total_time = time.time() - start
    logger.info(f"Dataset: {data_path}")
    logger.info(f"Average Time for 1 question: {total_time / len(questions)}")
    with open(save_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
