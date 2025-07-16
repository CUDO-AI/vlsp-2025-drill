from typing import List
import requests
import json
from tqdm import tqdm

def search_law_ids(api_key: str, law_ids: List[str], output_path: str = "data/search_result.json"):
    url = "https://google.serper.dev/search"

    headers = {
      'X-API-KEY': api_key,
      'Content-Type': 'application/json'
    }

    results = {}
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    except FileNotFoundError:
        pass

    for law_id in tqdm(law_ids, desc="Searching..."):
        law_id = law_id.strip()
        if law_id in results:
            continue

        payload = json.dumps({
            "q": law_id,
            "location": "Vietnam",
            "gl": "vn",
            "hl": "vi"
        })

        response = requests.post(url, headers=headers, data=payload)

        try:
            data = response.json()
        except Exception as e:
            print(f"❌ Lỗi khi parse JSON cho {law_id}: {e}")
            data = {"error": str(e), "status_code": response.status_code}

        results[law_id] = data

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)