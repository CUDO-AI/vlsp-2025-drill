import unicodedata

CHAR_REPLACEMENTS = {
    "Ð": "Đ",
    "ð": "đ",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "–": "-",
    "—": "-",
    # bạn có thể bổ sung thêm nếu cần
}


def normalize_general(text):
    # Bước 1: Unicode normalize
    text = unicodedata.normalize("NFKC", text)

    # Bước 2: Thay thế các ký tự dễ nhầm
    for bad, good in CHAR_REPLACEMENTS.items():
        text = text.replace(bad, good)

    return text