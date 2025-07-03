def calculate_precision(true_positives: int, false_positives: int) -> float:
    return true_positives / (true_positives + false_positives)


def calculate_recall(true_positives: int, false_negatives: int) -> float:
    return true_positives / (true_positives + false_negatives)


def calculate_f2(precision: float, recall: float) -> float:
    return (5 * precision * recall) / (4 * precision + recall + 1e-20)


# def compute_metrics(predictions: List[List[int]], growth_truth: List[int], save_path: str):
#     pass