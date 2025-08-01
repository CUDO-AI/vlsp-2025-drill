import json
import zipfile


def strategy_1(results_with_scores: dict):
    filtered_results = []
    for result in results_with_scores:
        scores = result['scores']
        filtered_ids = []
        for idx, score in enumerate(scores[:3]):
            if score > 0.99:
                filtered_ids.append(idx)
        if len(filtered_ids) == 0:
            filtered_ids = [0]
        result['relevant_laws'] = [result['relevant_laws'][idx] for idx in filtered_ids]
        result.pop('scores')
        filtered_results.append(result)
    return filtered_results


with open("results_with_scores.json", "r", encoding="utf-8") as f_in:
    results_with_scores = json.load(f_in)
    
filtered_results = strategy_1(results_with_scores)

with open("results.json", "w", encoding="utf-8") as f_out:
    json.dump(filtered_results, f_out, ensure_ascii=False, indent=4)
    
with zipfile.ZipFile("results.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
    zipf.write("results.json")