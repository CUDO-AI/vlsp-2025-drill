import json
from transformers import AutoTokenizer
from tqdm import tqdm
import random


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")

MAX_LENGTH = 1024

print("Loading datasets...")
with open("evaluation-result/bge_m3_ranker_v2_metrics_v4.json", "r", encoding="utf-8") as f_in:
    result = json.load(f_in)

with open("ir-datasets/v2/corpus.json", "r", encoding="utf-8") as f_in:
    corpus = json.load(f_in)
    
converted_corpus = {str(doc["id"]): doc["title"] + "\n" + doc["content"] for doc in corpus}


print("Mining hard negatives...")
mining_result = []
for sample in tqdm(result):
    question = sample["question"]
    relevants = sample["relevants"]
    positives = []
    for relevant in relevants:
        relevant_text = converted_corpus[relevant]
        tokens = tokenizer.tokenize(relevant_text)
        if len(tokens) > MAX_LENGTH:
            continue
        else:
            positives.append({
                'id': relevant,
                'text': relevant_text
            })
    if len(positives) == 0:
        continue
    rerank_hits = sample["rerank_hits"]
    hard_negatives = []
    count_negatives = 0
    for hit in rerank_hits:
        if count_negatives >= 15:
            break
        if hit not in relevants:
            hit_text = converted_corpus[hit]
            tokens = tokenizer.tokenize(hit_text)
            if len(tokens) > MAX_LENGTH:
                continue
            else:
                hard_negatives.append({
                    'id': hit,
                    'text': hit_text
                })
                count_negatives += 1
    if len(hard_negatives) == 0:
        continue
    else:
        # hard_negatives = random.sample(hard_negatives[5:], 5)
        hard_negatives = random.sample(hard_negatives, 5)
        mining_result.append({
            'question': question,
            'positives': positives,
            'hard_negatives': hard_negatives
        })

with open("mining_result_v2.json", "w", encoding="utf-8") as f_out:
    json.dump(mining_result, f_out, ensure_ascii=False, indent=4)