import logging
import traceback

import torch
import json
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split

from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderModelCardData
from sentence_transformers.cross_encoder.evaluation import (
    CrossEncoderRerankingEvaluator,
)
from sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
from sentence_transformers.evaluation.SequentialEvaluator import SequentialEvaluator

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def main():
    model_name = "BAAI/bge-m3"

    train_batch_size = 8
    num_epochs = 5

    # 1a. Load a model to finetune with 1b. (Optional) model card data
    model = CrossEncoder( model_name)
    logging.info(f"Model max length: {model.max_length}")
    logging.info(f"Model num labels: {model.num_labels}")

    # 2a. Load the VLSP-Drill dataset
    full_dataset = []
    with open("ir-datasets/fine-tuning/finetuning_data_v2.jsonl", "r", encoding="utf-8") as f_in:
        for line in f_in:
            full_dataset.append(json.loads(line))
            
    logging.info("Read the VLSP-Drill training dataset")
    train_dataset, eval_dataset = train_test_split(full_dataset, test_size=0.2, random_state=12)
    formatted_train_dataset = []
    for sample in train_dataset:
        query = sample["query"]
        pos = sample["pos"]
        neg = sample["neg"]
        for p in pos:
            formatted_train_dataset.append({
                "query": query,
                "passage": p,
                "label": 1
            })
        for n in neg:
            formatted_train_dataset.append({
                "query": query,
                "passage": n,
                "label": 0
            })
    train_dataset = Dataset.from_list(formatted_train_dataset)
    logging.info(train_dataset)
    
    num_positive_samples = sum(1 for sample in train_dataset if sample["label"] == 1)
    num_negative_samples = sum(1 for sample in train_dataset if sample["label"] == 0)
        
    pos_weight = num_negative_samples / num_positive_samples
    logging.info(f"Pos weight: {pos_weight}")
    

    loss = BinaryCrossEntropyLoss(model=model, pos_weight=torch.tensor(pos_weight))
    
    eval_samples = [
            {
                "query": sample["query"],
                "positive": sample["pos"],
                "negative": sample["neg"],
            }
            for sample in eval_dataset
        ]
    logging.info(f"Number of eval samples: {len(eval_samples)}")
    reranking_evaluator = CrossEncoderRerankingEvaluator(
        samples=eval_samples,
        batch_size=train_batch_size,
        name="vlsp-drill-dev",
        always_rerank_positives=False,
    )

    # 4c. Combine the evaluators & run the base model on them
    evaluator = SequentialEvaluator([reranking_evaluator])
    evaluator(model)

    # 5. Define the training arguments
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    run_name = f"reranker-{short_model_name}-vlsp-drill-bce-v3"
    args = CrossEncoderTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        dataloader_num_workers=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_vlsp-drill-dev_ndcg@10",
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        logging_steps=200,
        logging_first_step=True,
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
        seed=12,
    )

    # 6. Create the trainer & start training
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

    # 7. Evaluate the final model, useful to include these in the model card
    evaluator(model)

    # 8. Save the final model
    final_output_dir = f"models/{run_name}/final"
    model.save_pretrained(final_output_dir)
    

if __name__ == "__main__":
    main()