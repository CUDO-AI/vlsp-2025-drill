import re
import numpy as np

def normalize_titles(titles):
    if isinstance(titles, np.ndarray):
        titles = titles.tolist()
    elif not isinstance(titles, list):
        raise TypeError("Input must be a list or numpy array of strings")
    cleaned = [re.sub(r'\s+', ' ', t).strip() for t in titles]
    return cleaned

def normalize_text(text):
    # Bỏ dấu câu, khoảng trắng, xuống dòng, tab, v.v.
    text = re.sub(r'\s+', '', text)           # Bỏ toàn bộ khoảng trắng
    text = re.sub(r'[^\w]', '', text)         # Bỏ ký tự không phải chữ/số
    return text.lower()                       # Chuyển về chữ thường

def is_text_similar(text1, text2):
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    # So sánh xem 1 trong 2 chuỗi thuộc vào chuỗi còn lại
    return norm1 in norm2 or norm2 in norm1