from tqdm.auto import tqdm
from dotenv import load_dotenv
import json
import time
import os
import argparse

from rerankers.nvidia_reranker import NVIDIARanker
from rerankers.tei_reranker import TEIRanker
from utils import load_corpus, load_config, load_json
from logger import logger
from eval_metrics import compute_metrics


load_dotenv()


RANKER = {
    'nvidia': NVIDIARanker(base_url=os.getenv("NV_RERANK_BASE_URL")),
    'tei': TEIRanker(base_url=os.getenv("TEI_RERANK_BASE_URL"))
}


def evaluate(args):
    config = load_config(args.config_path)
    logger.info(f"Config: {config}")
    eval_data, corpus = load_json(config['eval_path']), load_corpus(config['corpus_path'])
    start_time = time.time()
    logger.info("Inference")
    for sample in tqdm(eval_data):
        question = sample['question']
        hybrid_hits = sample['hybrid_hits'][:config['top_k_eval']]
        passages = [corpus[int(doc_id)] for doc_id in hybrid_hits]
        assert len(passages) == config['top_k_eval']
        rankings = RANKER[args.ranker_type].rerank(question, passages, config['model_name'])
        sample['rerank_hits'] = [hybrid_hits[i['index']] for i in rankings]
    logger.info(f"Avg time inference: {(time.time() - start_time) / len(eval_data)} seconds")

    with open(config['save_path'], "w", encoding="utf-8") as f_out:
        json.dump(eval_data, f_out, ensure_ascii=False, indent=4)

    logger.info("Computing metrics")
    rankings_with_hits = [sample['rerank_hits'] for sample in eval_data]
    gold_hits = [sample['answer_id'] for sample in eval_data]
    compute_metrics(predictions=rankings_with_hits, growth_truth=gold_hits, save_path=config['log_path'])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="configure/rerankers/scb/mistral_4b_v3.yml", type=str)
    parser.add_argument('--ranker_type', default="nvidia", type=str)
    input_args = parser.parse_args()
    evaluate(input_args)
