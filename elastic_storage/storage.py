from elasticsearch import Elasticsearch, helpers
from logger import logger
from tqdm.auto import tqdm

from elastic_storage.utils import bm25_preprocessing
from elastic_storage.fusion import weight_sum_score


class ElasticsearchStore:
    def __init__(self, elastic_url: str, username: str, password: str):
        self.elastic_url = elastic_url
        self.client = Elasticsearch(
            self.elastic_url, 
            basic_auth=(username, password),
            ssl_show_warn=False, 
            verify_certs=False,
            timeout=30,  # Increase timeout to 30 seconds
            max_retries=3,  # Add retry mechanism
            retry_on_timeout=True
        )
        logger.info(f'Client: {self.client.info()}')

    def create_index(self, index: str, bm25_k1: float = 0.5, bm25_b: float = 0.5, 
                     dim: int = 768, m_hnsw: int = 32, ef_construction: int = 128, 
                     similarity: str = "dot_product"):
        """Create a hybrid elasticsearch index

        Args:
            m_hnsw: m_hnsw
            ef_construction: ef_construction
            index (str): name of index
            bm25_k1 (float, optional): k1 of bm25. Defaults to 0.5.
            bm25_b (float, optional): b of bm25. Defaults to 0.5.
            dim (int, optional): dimension of embedding model. Defaults to 1024.
            similarity (str, optional): similarity method. Defaults to "dot_product".
        """
        settings = {
            "similarity": {
                "bm25_similarity": {
                    "type": "BM25",
                    "k1": bm25_k1,
                    "b": bm25_b
                }
            }
        }
        mappings = {
            "_source": {
                "includes": [
                    "content",
                ],
                "excludes": [
                    "bm25_content", 
                    "embedding"
                ]
            },
            "properties": {
                "content": {
                    "type": "object", 
                    "enabled": "false"
                },
                "bm25_content": {
                    "type": "text",
                    "similarity": "bm25_similarity"
                },
                "embedding": {
                    "type": "dense_vector",
                    "dims": dim,
                    "index": "true",
                    "similarity": similarity,
                    "index_options": {
                        "type": "hnsw",
                        "m": m_hnsw,
                        "ef_construction": ef_construction
                    }
                }
            }
        }
                
        if self.client.indices.exists(index=index):
            message = f"#> Index '{index}' already exist."
            logger.warning(message)
            return {"status": "warning", "code": 400, "message": message}
        else:
            resp = self.client.indices.create(index=index, settings=settings, mappings=mappings)
            logger.info(resp)
            return {"status": "success", "code": 200, "message": resp}
        
    def is_index_exist(self, index: str):
        return self.client.indices.exists(index=index)
    
    def indexing(self, index: str, chunks: list[dict], batch_size: int = 512):
        """Adding chunks of a document to elasticsearch index
        Args:
            index (str): index want to add
            chunks (List): chunks 
            batch_size (int, optional): number of chunks for 1 add. Defaults to 1000.
        """
        if not self.client.indices.exists(index=index):
            logger.warning(f"#> Index '{index}' does not exist.")
            logger.info(f"#> Create new '{index}' index.")
            _ = self.create_index(index=index)
            
        with tqdm(total=len(chunks)) as pbar:
            for start_idx in range(0, len(chunks), batch_size):
                end_idx = start_idx + batch_size
                sub_chunks = chunks[start_idx:end_idx]
                bulk_data = self._prepare_bulk_data(chunks=sub_chunks)
                for data, idx in zip(bulk_data, range(start_idx, end_idx)):
                    data["_id"] = idx
                helpers.bulk(self.client, bulk_data, index=index)
                pbar.update(batch_size)
        message = "#> Save successfully."
        logger.info(message)
        self.client.indices.refresh(index=index)
        return {"status": "success", "code": 200, "message": message}

    def delete_index(self, index: str):
        if self.client.indices.exists(index=index):
            return self.client.indices.delete(index=index)
        else:
            logger.warning(f"#> Index '{index}' does not exist.")
    
    @staticmethod
    def _prepare_bulk_data(chunks: list[dict]) -> list[dict]:
        """Prepare bulk data for Elasticsearch indexing."""
        bulk_data = []
        for idx, chunk in enumerate(chunks):
            embedding = chunk.get("embedding", [])
            content = chunk.get("content", {})
            data = {
                "content": content
            }
            if embedding:
                title, context = content.get("title", ""), content.get("context", "")
                bm25_content = bm25_preprocessing(title + " " + context)

                data.update({
                    "embedding": embedding,
                    "bm25_content": bm25_content
                })
            bulk_data.append(data)

        return bulk_data

    def hybrid_search(self, index: str, query: str, query_embedding: list, top_k: int = 100):
        if self.client.indices.exists(index=index):
            bm25_query = bm25_preprocessing(query)
            # Define sparse_search query
            es_query = {
                "bool": {
                    "must": [
                        {"match": {"bm25_content": {"query": bm25_query}}}
                    ]
                }
            }
            # Define dense_search query 
            es_knn = {
                "field": "embedding", 
                "query_vector": query_embedding,
                "k": 100, 
                "num_candidates": 256
            }
            # Do sparse searching
            bm25_search = self.client.search(index=index, size=100, query=es_query, source=False)
            sparse_result = {hit["_id"]: hit["_score"] for hit in bm25_search["hits"]["hits"]}
            # Do dense searching
            dense_search = self.client.search(index=index, size=100, knn=es_knn, source=False)
            dense_result = {hit["_id"]: hit["_score"] for hit in dense_search["hits"]["hits"]}
            # Do hybrid searching
            hybrid_result = weight_sum_score(sparse_result, dense_result, top_k=top_k)
            # logger.info(f"#> Search complete.")
            return {"status": "success", "code": 200, "data": hybrid_result}
        else:
            message = f"#> Index '{index}' does not exist."
            logger.error(message)
            return {"status": "fail", "code": 400, "message": message}
        
    def sparse_search(self, index: str, query: str, top_k: int = 100):
        if self.client.indices.exists(index=index):
            bm25_query = bm25_preprocessing(query)
            # Define sparse_search query
            es_query = {
                "bool": {
                    "must": [
                        {"match": {"bm25_content": {"query": bm25_query}}}
                    ]
                }
            }
            # Do sparse searching
            bm25_search = self.client.search(index=index, query=es_query, source=False, size=top_k)
            sparse_result = {hit["_id"]: hit["_score"] for hit in bm25_search["hits"]["hits"]}
            # logger.info(f"#> Search complete.")
            return {"status": "success", "code": 200, "data": sparse_result}
        else:
            message = f"#> Index '{index}' does not exist."
            logger.error(message)
            return {"status": "fail", "code": 400, "message": message}

    def dense_search(self, index: str, query_embedding: list, top_k: int = 100, es_knn_top_k: int = 100):
        if self.client.indices.exists(index=index):
            # Define dense_search query
            es_knn = {
                "field": "embedding", 
                "query_vector": query_embedding,
                "k": es_knn_top_k, 
                "num_candidates": 256
            }
            # Do dense searching
            dense_search = self.client.search(index=index, size=top_k, knn=es_knn, source=False)
            dense_result = {hit["_id"]: hit["_score"] for hit in dense_search["hits"]["hits"]}
            return {"status": "success", "code": 200, "data": dense_result}
        else:
            message = f"#> Index '{index}' does not exist."
            logger.error(message)
            return {"status": "fail", "code": 400, "message": message}