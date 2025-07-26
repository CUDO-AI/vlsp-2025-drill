import json
from collections import Counter, defaultdict
from bs4 import BeautifulSoup


# Đường dẫn file json
JSON_PATH = 'law_contents.json'

# Đọc file json lớn theo từng dòng (nếu là jsonlines) hoặc toàn bộ (nếu là list)
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception:
            # Nếu là jsonlines
            data = [json.loads(line) for line in f]
    return data


def extract_a_names(html):
    """Trích xuất tất cả thuộc tính name của thẻ <a> trong html"""
    soup = BeautifulSoup(html, 'html.parser')
    return [a.get('name') for a in soup.find_all('a', attrs={'name': True})]


def main():
    data = load_json(JSON_PATH)
    name_counter = Counter()
    prefix_counter = Counter()
    prefix_examples = defaultdict(list)

    for doc in data:
        # Đoán trường content chứa html
        html = doc.get('content_html') or doc.get('html') or ''
        names = extract_a_names(html)
        for name in names:
            name_counter[name] += 1
            # Lấy prefix (phần trước dấu _ nếu có)
            prefix = name.split('_')[0] if '_' in name else name
            prefix_counter[prefix] += 1
            if len(prefix_examples[prefix]) < 10:
                prefix_examples[prefix].append(name)

    print('Thống kê các tiền tố name xuất hiện trong thẻ <a> (top 20):')
    for prefix, count in prefix_counter.most_common(20):
        print(f'- {prefix}: {count} lần. Ví dụ: {prefix_examples[prefix]}')

    print('\nTop 20 giá trị name phổ biến nhất:')
    for name, count in name_counter.most_common(20):
        print(f'- {name}: {count} lần')

if __name__ == '__main__':
    main() 