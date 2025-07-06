from core.storage.elasticsearch.storage import ElasticsearchStore
from core.encoders.ncp import NCPEncoder
from core.evaluation.retrieval.evaluate import evaluate as retrieve_eval
from indexing import indexing

import argparse
from dotenv import load_dotenv
import os
import yaml


load_dotenv()


ES_STORE = ElasticsearchStore(elastic_url=os.getenv("ELASTIC_URL"), 
                              username=os.getenv("ES_USERNAME"), 
                              password=os.getenv("ES_PASSWORD"))

NCP_EMBEDDER = NCPEncoder(base_url=os.getenv("NCP_EMBEDDING_BASE_URL"),
                            api_key=os.getenv("NCP_EMBEDDING_API_KEY"),
                            model_name=os.getenv("NCP_EMBEDDING_MODEL_NAME"))

def evaluate(args):
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    indexing_params = {
        'es_store': ES_STORE,
        "index": config['index'],
        "corpus_path": config['corpus_path'],
        "embedding_path": config['embedding_path'],
        "dim": config['dimension'],
        "overwrite_index": config['overwrite_index'],
        
    }
    indexing(**indexing_params)
    
    NCP_EMBEDDER.model_name = config['model_name']
    eval_params = {
        'es_store': ES_STORE,
        'embedder': NCP_EMBEDDER,
        'eval_method': config['eval_method'],
        'data_path': config['data_path'],
        'index': config['index'],
        'log_path': config['log_path'],
        'dimension': config['dimension'],
        'save_path': config['save_path']
        
    }
    retrieve_eval(**eval_params)
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="configure/retrievers/vlsp/vlsp_ncp_e5_large.yml", type=str)
    input_args = parser.parse_args()
    evaluate(input_args)