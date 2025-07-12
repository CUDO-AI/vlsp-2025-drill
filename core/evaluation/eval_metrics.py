from typing import List, Dict, Any
import csv
import os


def calculate_precision_at_k(predictions: List[List[str]], ground_truth: List[List[str]], k: int) -> float:
    """
    Tính precision@k cho bài toán retrieval.
    
    Args:
        predictions: List các list chứa doc_ids được retrieve cho mỗi query
        ground_truth: List các list chứa doc_ids đúng cho mỗi query
        k: Số lượng documents top-k để tính precision
    
    Returns:
        float: Giá trị precision@k trung bình
    """
    assert len(predictions) == len(ground_truth), "Số lượng predictions và ground_truth phải bằng nhau"
    
    total_precision = 0.0
    
    for pred, gt in zip(predictions, ground_truth):
        assert len(gt) > 0, "Ground truth không được rỗng"
            
        # Lấy top-k predictions
        top_k_pred = pred[:k]
        
        # Đếm số relevant documents trong top-k
        relevant_in_top_k = len(set(top_k_pred) & set(gt))
        
        # Tính precision@k cho query này
        precision_at_k = relevant_in_top_k / len(top_k_pred) if len(top_k_pred) > 0 else 0.0
        
        total_precision += precision_at_k
    
    return total_precision / len(predictions)


def calculate_recall_at_k(predictions: List[List[str]], ground_truth: List[List[str]], k: int) -> float:
    """
    Tính recall@k cho bài toán retrieval.
    
    Args:
        predictions: List các list chứa doc_ids được retrieve cho mỗi query
        ground_truth: List các list chứa doc_ids đúng cho mỗi query
        k: Số lượng documents top-k để tính recall
    
    Returns:
        float: Giá trị recall@k trung bình
    """
    assert len(predictions) == len(ground_truth), "Số lượng predictions và ground_truth phải bằng nhau"
    
    total_recall = 0.0
    
    for pred, gt in zip(predictions, ground_truth):
        if len(gt) == 0:  # Skip queries không có relevant documents
            continue
            
        # Lấy top-k predictions
        top_k_pred = pred[:k]
        
        # Đếm số relevant documents trong top-k
        relevant_in_top_k = len(set(top_k_pred) & set(gt))
        
        # Tính recall@k cho query này
        recall_at_k = relevant_in_top_k / len(gt) if len(gt) > 0 else 0.0
        
        total_recall += recall_at_k
    
    return total_recall / len(predictions)


def calculate_metrics_at_k(predictions: List[List[str]], 
                           ground_truth: List[List[str]], 
                           k_values: List[int] = [1, 3, 5, 10], 
                           save_path: str = "") -> Dict[str, Any]:
    """
    Tính các metrics retrieval cho nhiều giá trị k khác nhau.
    
    Args:
        predictions: List các list chứa doc_ids được retrieve cho mỗi query
        ground_truth: List các list chứa doc_ids đúng cho mỗi query
        k_values: List các giá trị k để tính metrics (mặc định [1, 3, 5, 10])
        save_path: Đường dẫn file CSV để lưu kết quả (mặc định: '', không lưu)
    
    Returns:
        Dict chứa các metrics: precision@k, recall@k, f1@k, f2@k
    """
    results = {}
    rows = []
    
    for k in k_values:
        precision_k = calculate_precision_at_k(predictions, ground_truth, k)
        recall_k = calculate_recall_at_k(predictions, ground_truth, k)
        mrr_k = calculate_mrr_at_k(predictions, ground_truth, k)
        # Tính F1@k
        f1_k = 0.0
        f2_k = 0.0
        if precision_k + recall_k > 0:
            f1_k = 2 * precision_k * recall_k / (precision_k + recall_k)
            f2_k = 5 * precision_k * recall_k / (4 * precision_k + recall_k)
        
        results[f'precision@{k}'] = precision_k
        results[f'recall@{k}'] = recall_k
        results[f'f1@{k}'] = f1_k
        results[f'f2@{k}'] = f2_k
        results[f'mrr@{k}'] = mrr_k
        
        rows.append([f"Top_{k}", precision_k, recall_k, f1_k, f2_k, mrr_k])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_path:
        with open(save_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Top_k', 'Precision@K', 'Recall@K', 'F1@K', 'F2@K', 'MRR@K'])
            for row in rows:
                writer.writerow(row)
    
    return results


def calculate_mrr_at_k(predictions: List[List[str]], ground_truth: List[List[str]], k: int) -> float:
    """
    Tính Mean Reciprocal Rank (MRR).
    
    Args:
        predictions: List các list chứa doc_ids được retrieve cho mỗi query
        ground_truth: List các list chứa doc_ids đúng cho mỗi query
    
    Returns:
        float: Giá trị MRR
    """
    assert len(predictions) == len(ground_truth), "Số lượng predictions và ground_truth phải bằng nhau"
    
    total_reciprocal_rank = 0.0
    
    for pred, gt in zip(predictions, ground_truth):
        if len(gt) == 0:
            continue
            
        # Tìm vị trí đầu tiên của relevant document
        for rank, doc_id in enumerate(pred[:k], 1):
            if doc_id in gt:
                total_reciprocal_rank += 1.0 / rank
                break
        else:
            # Không tìm thấy relevant document nào
            total_reciprocal_rank += 0.0
            
    return total_reciprocal_rank / len(predictions)
