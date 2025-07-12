from core.storage.elasticsearch.storage import ElasticsearchStore
from core.encoders.bge import BGEEncoder
from core.evaluation.retrieval.evaluate import evaluate as retrieve_eval
from indexing import indexing
from core.logger import logger

import argparse
from dotenv import load_dotenv
import os
import yaml


load_dotenv()


def evaluate(args):
    logger.info("Initializing Elasticsearch store")
    ES_STORE = ElasticsearchStore(
        elastic_url=os.getenv("ELASTIC_URL", ""), 
        username=os.getenv("ES_USERNAME", ""), 
        password=os.getenv("ES_PASSWORD", "")
    )
    
    logger.info("Loading config")
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    logger.info(config)
    logger.info("Indexing")
    indexing_params = {
        'es_store': ES_STORE,
        "index": config['index'],
        "corpus_path": config['corpus_path'],
        "embedding_path": config['embedding_path'],
        "dim": config['dimension'],
        "overwrite_index": config['overwrite_index'],
        
    }
    
    indexing(**indexing_params)

    logger.info("Initializing BGE embedder")
    BGE_EMBEDDER = BGEEncoder(model_name_or_path=config['model_name'], 
                              prefix_query=config['prefix_query'], 
                              prefix_passage=config['prefix_passage'])
    
    logger.info("Evaluating")
    eval_params = {
        'es_store': ES_STORE,
        'embedder': BGE_EMBEDDER,
        'eval_method': config['eval_method'],
        'data_path': config['data_path'],
        'index': config['index'],
        'log_path': config['log_path'],
        'dimension': config['dimension'],
        'save_path': config['save_path']
        
    }
    retrieve_eval(**eval_params)

    logger.info("Done")
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="configure/me5_v1.yml", type=str)
    input_args = parser.parse_args()
    evaluate(input_args)