from core.storage.elasticsearch.storage import ElasticsearchStore
from core.logger import logger
from core.encoders.base import BaseEncoder
from core.datasets.data_loader import load_test_data

from tqdm.auto import tqdm
import numpy as np
import time
import csv
import json


SLEEP_TIME = 20


def _get_hits(es_store: ElasticsearchStore, embedder: BaseEncoder,
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
    logger.info('Example for question: ', questions[:5])
    assert len(answer_ids) == len(questions)
    logger.info('Starting evaluate...!')
    counts_recall = np.zeros((6, ))
    counts_mrr = np.zeros((6, ))
    report_top_k = [1, 3, 5, 10, 20, 50]
    start = time.time()
    result = []
    count = 0
    for question, answer_id in tqdm(zip(questions, answer_ids), total=len(answer_ids)):
        if count > 1000:
            time.sleep(SLEEP_TIME)
            logger.info(f"Sleeping for {SLEEP_TIME} seconds...")
            count = 0
        hits = _get_hits(es_store, embedder, eval_method, index, question, dimension)
        result.append({"question": question, "hits": hits})       
        hit_indices = [int(hit) == int(answer_id) for hit in hits]
        if any(hit_indices):
            idx = hit_indices.index(True)
            for i, r in enumerate(report_top_k):
                if idx < r:
                    counts_recall[i:] += 1
                    counts_mrr[i:] += 1.0 / (idx + 1)
                    break
        count += 1
    total_time = time.time() - start
    logger.info(f"Dataset: {data_path}")
    logger.info(f"Average Time for 1 question: {total_time / len(questions)}")
    header = ["Top K", "Recall@K", "MRR@K"]
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx, k in enumerate(report_top_k):
            recall = counts_recall[idx] / len(questions) * 100
            mrr = counts_mrr[idx] / len(questions) * 100
            logger.info(f"Recall@{k}: {recall}")
            logger.info(f"MRR@{k}: {mrr}")
            writer.writerow([str(k), str(recall), str(mrr)])
        writer.writerow(["Average time", str(total_time / len(questions)), ""])
    with open(save_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
