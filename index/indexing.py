from elastic_storage.storage import ElasticsearchStore
from logger import logger
import numpy as np
import json


def convert_chunks(corpus_path: str, embedding_path: str):
    with open(corpus_path) as f:
        corpus_chunks = json.load(f)
    corpus_embeddings = np.load(embedding_path)
    corpus_embeddings = corpus_embeddings.tolist()
    formatted_chunks = []
    for chunk, embedding in zip(corpus_chunks, corpus_embeddings):
        if isinstance(chunk, str):
            title, context = "", chunk
        else:
            title, context = chunk["passage_title"], chunk["passage_content"]
        formatted_chunks.append(
            {
                "content": {
                    "title": title,
                    "context": context
                },
                "embedding": embedding
            }
        )
    return formatted_chunks


def indexing(es_store: ElasticsearchStore, index: str, corpus_path: str, 
             embedding_path: str, dim: int = 1024, overwrite_index: bool = False):
    formatted_chunks = convert_chunks(corpus_path, embedding_path)
    if es_store.is_index_exist(index):
        if overwrite_index:
            es_store.delete_index(index=index)
            es_store.create_index(index=index, dim=dim)
            es_store.indexing(index=index, chunks=formatted_chunks)
        else:
            logger.info(f"Index '{index}' already exists. Skip indexing.")
    else:
        es_store.create_index(index=index, dim=dim)
        es_store.indexing(index=index, chunks=formatted_chunks)
