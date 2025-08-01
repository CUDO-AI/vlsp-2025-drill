import json

with open("ir-datasets/mining_result_v2.json", "r", encoding="utf-8") as f_in:
    mining_result = json.load(f_in)

with open("ir-datasets/fine-tuning/finetuning_data_v2.jsonl", "w", encoding="utf-8") as f_out:
    for sample in mining_result:
        finetuning_sample = {
            "query": sample["question"],
            "pos": [positive["text"] for positive in sample["positives"]],
            "neg": [hard_negative["text"] for hard_negative in sample["hard_negatives"]]
        }
        f_out.write(json.dumps(finetuning_sample, ensure_ascii=False) + "\n")