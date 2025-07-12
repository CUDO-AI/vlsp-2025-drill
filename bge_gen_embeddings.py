from core.encoders.bge import BGEEncoder
from core.logger import logger
from core.datasets.data_loader import load_corpus

import argparse
import numpy as np


def gen_embeddings(model_name_or_path: str, data_path: str, save_path: str, prompt_name: str):
    logger.info("Loading corpus")
    corpus = load_corpus(data_path)
    logger.info("Loading model")
    model = BGEEncoder(model_name_or_path=model_name_or_path)
    logger.info("Generating embeddings")
    embeddings = model.encode_passages(corpus, prompt_name=prompt_name, max_length=1024)
    logger.info("Saving embeddings")
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)
    np.save(save_path, embeddings)
    logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="BAAI/bge-m3", type=str)
    parser.add_argument("--prompt_name", default="", type=str)
    parser.add_argument("--data_path", default="ir-datasets/reformatted/corpus.json", type=str)
    parser.add_argument("--save_path", default="embeddings/bge_m3_v1.npy", type=str)
    args = parser.parse_args()
    logger.info(args)
    gen_embeddings(args.model_name_or_path, args.data_path, args.save_path, args.prompt_name)
