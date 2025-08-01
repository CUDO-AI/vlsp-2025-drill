import json
import random

random.seed(12)


formatted_data = []
data = []
with open("ir-datasets/fine-tuning/finetuning_data_v1.jsonl", "r", encoding="utf-8") as f_in:
    for line in f_in:
        data.append(json.loads(line))

for sample in data:
    query = sample["query"]
    pos = sample["pos"]
    neg = sample["neg"]
    for p in pos:
        formatted_data.append({
            "query": query,
            "passage": p,
            "label": 1
        })
    for n in neg:
        formatted_data.append({
            "query": query,
            "passage": n,
            "label": 0
        })

with open("ir-datasets/fine-tuning/finetuning_sbert_data_v1.json", "w") as f_out:
    json.dump(formatted_data, f_out, ensure_ascii=False, indent=4)