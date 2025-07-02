import re
import string
import unicodedata as ud
from pyvi import ViTokenizer


puncts = string.punctuation + ''.join(['...', '....', '–', '●', '•', "→", "…", "….", "✓"])

replace_phrase = {'\n': ' ', "‘": "'", "”": '"', "“": '"', "′": "'", "``": "", "''": "", "|": ""}

vi_stopwords = ["như", "làm", "là", "và", "với", "nếu", "thì", "do", "ở", "đây", "đó", "lại", "không", "nhỉ", "ta",
                "cho", "chung", "đã", "nơi", "để", "đến", "số", "một", "khác", "được", "vào", "ra", "trong", "ạ",
                "người", "loài", "từ", "nào", "bằng", "rằng", "nên", "gì", "việc", "ấy", "khi", "này", "chỉ", "về",
                "các", "còn", "trên", "những", "có", "mà", "nhưng", "nhiều", "nó", "sẽ", "chưa", "lúc", "có_thể",
                "bởi_vì", "tại_vì", "như_thế", "thế_là", "trong_khi", "thế_mà", "chẳng_hạn", "do_đó", "tuy_nhiên",
                "đôi_khi", "chỉ_là", "một_số", "chúng_nó", "rằng_là", "tôi", "năm"]


def normalize(text: str) -> str:
    """Normalize text"""
    text = ud.normalize("NFC", text)
    for phrase in replace_phrase:
        text = text.replace(phrase, replace_phrase[phrase])
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def lower_case(word: str) -> str:
    """Lowercase single word"""
    return word.lower()


def is_not_punctuation(word: str) -> bool:
    """Check if word is puncts"""
    return word not in puncts


def is_not_stop_word(word: str) -> bool:
    """Check if word is stopword"""
    return word not in vi_stopwords


def bm25_preprocessing(text: str) -> str:
    """Pre-processing input for bm25 search"""
    text = normalize(text)
    # Use pyvi for vietnamese word tokenize
    tokens = [lower_case(token) for token in ViTokenizer.tokenize(text).split()]
    # Remove word if is puncts or stopword
    tokens = [token for token in tokens if is_not_punctuation(token) and is_not_stop_word(token)]
    
    return normalize(" ".join(tokens))


def dense_preprocessing(text: str, prefix: str = "") -> str:
    """Pre-processing input for dense search"""
    text = normalize(text)
    if prefix:
        text = f"{prefix} {text}"
        
    return text
