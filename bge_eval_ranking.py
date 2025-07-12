from tqdm.auto import tqdm
from dotenv import load_dotenv
import json
import time
import argparse

from core.rerankers.bge import BGEReranker
from core.datasets.data_loader import load_rank_corpus, load_config, load_json
from core.logger import logger
from core.evaluation.eval_metrics import calculate_metrics_at_k


load_dotenv()


def evaluate(args):
    config = load_config(args.config_path)
    logger.info(f"Config: {config}")
    logger.info("Initializing BGE reranker")
    RANKER = BGEReranker(model_name_or_path=config['model_name'])
    logger.info("Loading eval data")
    eval_data, corpus = load_json(config['eval_path']), load_rank_corpus(config['corpus_path'])
    start_time = time.time()
    logger.info("Inference")
    for sample in tqdm(eval_data):
        question = sample['question']
        passages = [corpus[doc_id] for doc_id in list(sample['retrieval_hits'].keys())]
        rankings = RANKER.rerank(question, passages, config['batch_size'], config['max_length'])
        sample['rerank_hits'] = {str(psg["id"]): psg["score"] for psg in rankings}
    logger.info(f"Avg time inference: {(time.time() - start_time) / len(eval_data)} seconds")

    with open(config['save_path'], "w", encoding="utf-8") as f_out:
        json.dump(eval_data, f_out, ensure_ascii=False, indent=4)

    logger.info("Computing metrics")
    rankings_with_hits = [list(sample['rerank_hits'].keys()) for sample in eval_data]
    gold_hits = [sample['relevants'] for sample in eval_data]

    calculate_metrics_at_k(predictions=rankings_with_hits, ground_truth=gold_hits, save_path=config['log_path'])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="configure/bge_ranker_m3_v2.yml", type=str)
    parser.add_argument('--ranker_type', default="nvidia", type=str)
    input_args = parser.parse_args()
    evaluate(input_args)
