from core.evaluation.retrieval.evaluate import search
from core.encoders.bge import BGEEncoder
from core.rerankers.bge import BGEReranker
from core.rerankers.bge_layerwise import BGELayerWiseReranker
from core.storage.elasticsearch.storage import ElasticsearchStore
from core.datasets.data_loader import load_rank_corpus


import os
import json
import zipfile
from tqdm import tqdm
from dotenv import load_dotenv


load_dotenv()

# 0.71 -> Vietnamese Reranker
# 0.53 -> models/reranker-bge-reranker-v2-m3-vlsp-drill-bce-v2/final
# 

ES_STORE = ElasticsearchStore(
    elastic_url=os.getenv("ELASTIC_URL", ""), 
    username=os.getenv("ES_USERNAME", ""), 
    password=os.getenv("ES_PASSWORD", "")
    )


EMBEDDER = BGEEncoder(model_name_or_path="intfloat/multilingual-e5-large")
# RANKER = BGEReranker(model_name_or_path="BAAI/bge-reranker-v2-m3")
RANKER = BGELayerWiseReranker(model_name_or_path="BAAI/bge-reranker-v2-minicpm-layerwise")


with open("ir-datasets/public_test.json", "r", encoding="utf-8") as f_in:
    queries = json.load(f_in)
corpus = load_rank_corpus("ir-datasets/v2/corpus.json")
results = []
results_with_scores = []
top_k = 20
top_k_submit = 3
for query in tqdm(queries):
    question = query["question"]
    hits = search(ES_STORE, EMBEDDER, "hybrid", "me5_v2", question, 1024)
    passages = [corpus[doc_id] for doc_id in list(hits.keys())]
    rankings = RANKER.rerank(question, passages, 32, 1024)
    aids = [int(psg['id']) for psg in rankings[:top_k]]
    results.append({
        "qid": query['id'],
        "relevant_laws": aids[:top_k_submit]
    })
    results_with_scores.append({
        "qid": query['id'],
        "relevant_laws": aids,
        "scores": [psg['score'] for psg in rankings[:top_k]]
    })

with open("results.json", "w", encoding="utf-8") as f_out:
    json.dump(results, f_out, ensure_ascii=False, indent=4)

with open("results_with_scores.json", "w", encoding="utf-8") as f_out:
    json.dump(results_with_scores, f_out, ensure_ascii=False, indent=4)

# Create a zip file containing the results.json
with zipfile.ZipFile("results.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
    zipf.write("results.json")

print("Results saved to results.json and compressed to results.zip")