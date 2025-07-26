import cloudscraper
import json
from tqdm import tqdm
import time

output_file = 'law_contents.json'

def check_need_crawl(data):
    content_html = data.get('content_html', '')
    if content_html and "Just a moment..." not in content_html:
        return False
    return True

with open('law_contents.json', 'r') as f:
    data = json.load(f)

scraper = cloudscraper.create_scraper()

need_crawl = [law for law in data if check_need_crawl(law)]

print(f'Need to crawl {len(need_crawl)} laws')

error_count = 0

for law in tqdm(need_crawl, desc='Crawling HTML'):
    law_id = law['law_id']
    url = law['url']
    try:
        response = scraper.get(url)
        content = response.text
        if "Just a moment..." in content:
            error_count += 1
            law['content_html'] = ''
            law['error'] = 'Just a moment...'
            time.sleep(10)
            continue
        law['content_html'] = content
    except Exception as e:
        error_count += 1
        law['content_html'] = ''
        law['error'] = str(e)

with open(output_file, 'w') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f'Error count: {error_count}')