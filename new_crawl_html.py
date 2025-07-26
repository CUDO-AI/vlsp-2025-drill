import json
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

output_file = 'law_contents.json'

def check_need_crawl(data):
    content_html = data.get('content_html', '')
    if content_html and "Just a moment..." not in content_html:
        return False
    return True

with open('law_contents.json', 'r') as f:
    data = json.load(f)

chrome_options = Options()
chrome_options.add_argument("--headless")  # Chạy không hiện cửa sổ trình duyệt
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

need_crawl = [law for law in data if check_need_crawl(law)]

print(f'Need to crawl {len(need_crawl)} laws')

for law in tqdm(need_crawl, desc='Crawling HTML'):
    law_id = law['law_id']
    url = law['url']
    try:
        driver.get(url)
        time.sleep(5)  # Đợi trang load và vượt qua Cloudflare (có thể tăng lên nếu cần)
        content = driver.page_source

        if "Just a moment..." in content or content == '':
            print(f"Cloudflare challenge at {url}")
            law['content_html'] = ''
            law['error'] = 'Cloudflare challenge'
        else:
            law['content_html'] = content
    except Exception as e:
        print(f'Error crawling {law_id}: {str(e)}')
        law['content_html'] = ''
        law['error'] = str(e)

driver.quit()

with open(output_file, 'w') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)