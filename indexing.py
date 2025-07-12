from core.storage.elasticsearch.storage import ElasticsearchStore
from core.logger import logger
import numpy as np
import json


def convert_corpus(corpus_path: str, embedding_path: str):
    with open(corpus_path) as f:
        corpus = json.load(f)
    corpus_embeddings = np.load(embedding_path)
    corpus_embeddings = corpus_embeddings.tolist()
    formatted_chunks = []
    for doc, embedding in zip(corpus, corpus_embeddings):
        if isinstance(doc, str):
            raise ValueError("Corpus is a string")
        else:
            title, content = doc["title"], doc["content"]
        formatted_chunks.append(
            {
                "id": str(doc["id"]),
                "content": {
                    "title": title,
                    "content": content
                },
                "embedding": embedding
            }
        )
    return formatted_chunks


def indexing(es_store: ElasticsearchStore, index: str, corpus_path: str, 
             embedding_path: str, dim: int = 1024, overwrite_index: bool = False):
    formatted_corpus = convert_corpus(corpus_path, embedding_path)
    if es_store.is_index_exist(index):
        if overwrite_index:
            es_store.delete_index(index=index)
            es_store.create_index(index=index, dim=dim)
            es_store.indexing(index=index, chunks=formatted_corpus)
        else:
            logger.info(f"Index '{index}' already exists. Skip indexing.")
    else:
        es_store.create_index(index=index, dim=dim)
        es_store.indexing(index=index, chunks=formatted_corpus)
