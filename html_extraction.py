from bs4 import BeautifulSoup
import json
from tqdm import tqdm
import logging

def get_deepest(div):
    # Tìm tất cả div con trực tiếp
    child_divs = div.find_all('div', recursive=False)
    if not child_divs:
        return div
    # Nếu có div con, đi sâu vào div đầu tiên (hoặc lặp qua tất cả nếu muốn lấy nhiều nhất)
    return get_deepest(child_divs[0])

def chunk_document(html):
    soup = BeautifulSoup(html, 'html.parser')
    container = soup.find('div', class_='content1')
    if not container:
        return []
    container = get_deepest(container)
    # Định nghĩa thứ tự các cấp tiêu đề
    levels   = ['loai', 'chuong', 'muc', 'dieu', 'khoan']
    current  = { lvl: None for lvl in levels }
    chunks   = []
    buffer   = []

    def close_chunk():
        # Chỉ đóng chunk khi đã gặp loai_… (seen_loai) và buffer không rỗng
        if buffer:
            titles = [current[l] for l in levels if current[l]]
            chunks.append({
                'titles': titles,
                'content': '\n'.join(buffer).strip()
            })
        buffer.clear()
    
    for tag in container.find_all(recursive=False):
        check_flag = False
        first_a = tag.find('a')
        if first_a and first_a.has_attr('name'):
            # Nếu thẻ <a> có thuộc tính name, thì đó là tiêu đề
            name = first_a['name']
            for lvl in levels:
                if name.startswith(lvl):
                    check_flag = True
                    if name.endswith("_name"):
                        if current[lvl] is None:
                            current[lvl] = first_a.get_text(strip=True)
                        else:
                            current[lvl] += " " + first_a.get_text(strip=True)
                    else:
                        close_chunk()
                        idx = levels.index(lvl)
                        # reset các cấp thấp hơn
                        for lower in levels[idx+1:]:
                            current[lower] = None
                        # lấy text làm title
                        current[lvl]  = first_a.get_text(strip=True)
                        break
            if not check_flag:
                text = tag.get_text(separator=' ', strip=True)
                if text:
                    buffer.append(text)
        else:
            text = tag.get_text(separator=' ', strip=True)
            if text:
                buffer.append(text)
    close_chunk()
    return chunks


with open('law_contents.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

error_items = []

for item in tqdm(data, total=len(data), desc="Processing documents"):
    content = item['content_html'].strip()
    chunks = chunk_document(content)
    if not chunks:
        error_items.append(item)
        logging.warning(f'No chunks found in {item["law_id"]}')
    item['chunks'] = chunks

with open('law_content_chunks.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

with open('error_items.json', 'w', encoding='utf-8') as f:
    json.dump(error_items, f, ensure_ascii=False, indent=4)