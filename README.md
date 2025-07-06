# VLSP 2025 DRiLL Retrieval Task

Dự án thực hiện bài toán retrieval cho cuộc thi VLSP 2025 DRiLL.

## Setup

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Cấu hình biến môi trường

Tạo file `.env` trong thư mục gốc với nội dung:

```env
NCP_EMBEDDING_BASE_URL=https://mkp-api.fptcloud.com
NCP_EMBEDDING_API_KEY=your_api_key_here
NCP_EMBEDDING_MODEL_NAME=your_model_name_here
```

**Lưu ý**: Thay thế `your_api_key_here` và `your_model_name_here` bằng giá trị thực từ FPT Cloud.

### 3. Chuẩn bị dữ liệu

Đảm bảo có các file dữ liệu trong thư mục `ir-datasets/`:
- `ir-datasets/reformatted/corpus.json` - Corpus đã được format lại
- `ir-datasets/reformatted/queries.json` - Queries đã được format lại

## Sử dụng

### Test NCP API

Trước khi chạy các script chính, hãy test API:

```bash
python test_ncp_api.py
```

### Tạo embeddings cho corpus

```bash
python ncp_embedd.py
```

### Các script khác

- `reformat_data.py` - Format lại dữ liệu gốc
- `indexing.py` - Tạo index cho retrieval
- `ncp_eval_retrieval.py` - Đánh giá retrieval

## Cấu trúc dự án

```
vlsp-2025-drill/
├── core/                    # Core modules
│   ├── encoders/           # Embedding encoders
│   └── ...
├── ir-datasets/            # Dataset files
├── scripts/                # Utility scripts
├── .env                    # Environment variables (create this)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Troubleshooting

### Lỗi "ModuleNotFoundError: No module named 'src'"

Nếu gặp lỗi import, hãy chạy script từ thư mục gốc của dự án:

```bash
cd /path/to/vlsp-2025-drill
python script_name.py
```

### Lỗi "400 Bad Request" với NCP API

1. Kiểm tra API key có hợp lệ không
2. Kiểm tra model name có được hỗ trợ không
3. Chạy `test_ncp_api.py` để debug
4. Kiểm tra format của request trong debug logs