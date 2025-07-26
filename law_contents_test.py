import json
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from tqdm import tqdm

def create_driver():
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--window-size=1920,1080')
    return webdriver.Chrome(options=chrome_options)

# Đọc toàn bộ link từ law_urls.json
with open('law_urls.json', 'r', encoding='utf-8') as f:
    url_dict = json.load(f)

items = list(url_dict.items())

# Nếu đã có file kết quả, chỉ crawl lại các link bị lỗi
results = []
existing = {}
if os.path.exists('law_contents_test.json'):
    with open('law_contents_test.json', 'r', encoding='utf-8') as f:
        results = json.load(f)
    # Tạo dict để tra cứu nhanh
    for entry in results:
        existing[entry['code']] = entry
    # Chỉ crawl lại các link bị lỗi
    items_to_crawl = [(e['code'], e['url']) for e in results if e.get('error') or not e.get('content_html')]
    print(f'Found {len(items_to_crawl)} error cases, will retry only these.')
else:
    items_to_crawl = items

# Nếu không còn link nào cần crawl lại thì kết thúc
if not items_to_crawl:
    print('No error cases to retry. All done!')
    exit(0)

driver = create_driver()

for idx, (code, url) in enumerate(tqdm(items_to_crawl, desc='Retry crawling errors')):
    if idx > 0 and idx % 100 == 0:
        driver.quit()
        driver = create_driver()
    try:
        driver.get(url)
        time.sleep(4)
        try:
            content_div = driver.find_element(By.ID, 'tab1')
            content_html = content_div.get_attribute('outerHTML')
        except Exception:
            content_html = ''
        existing[code] = {
            'code': code,
            'url': url,
            'content_html': content_html
        }
    except Exception as e:
        # Nếu lỗi invalid session id, khởi tạo lại driver và thử lại 1 lần
        if 'invalid session id' in str(e):
            driver.quit()
            driver = create_driver()
            try:
                driver.get(url)
                time.sleep(4)
                try:
                    content_div = driver.find_element(By.ID, 'tab1')
                    content_html = content_div.get_attribute('outerHTML')
                except Exception:
                    content_html = ''
                existing[code] = {
                    'code': code,
                    'url': url,
                    'content_html': content_html
                }
                continue
            except Exception as e2:
                existing[code] = {
                    'code': code,
                    'url': url,
                    'content_html': '',
                    'error': str(e2)
                }
        else:
            existing[code] = {
                'code': code,
                'url': url,
                'content_html': '',
                'error': str(e)
            }
    time.sleep(1)
driver.quit()

# Gộp lại kết quả: giữ thứ tự gốc
final_results = []
for code, url in items:
    if code in existing:
        final_results.append(existing[code])
    else:
        final_results.append({'code': code, 'url': url, 'content_html': '', 'error': 'Not crawled'})

# Lưu kết quả ra file
with open('law_contents_test.json', 'w', encoding='utf-8') as f:
    json.dump(final_results, f, ensure_ascii=False, indent=2)

print('Done. Kết quả lưu ở law_contents_test.json')
# Hướng dẫn cài đặt:
# pip install selenium tqdm
# Tải chromedriver phù hợp với Chrome: https://chromedriver.chromium.org/downloads
# Đảm bảo chromedriver nằm trong PATH hoặc chỉ định đường dẫn trong script 