"""
Data Reformatter for VLSP 2025 DRiLL Retrieval Task

This module provides utilities to reformat the original dataset format
into a consistent format for retrieval tasks.
"""

import json
from typing import List, Dict, Any, Tuple
from pathlib import Path

from core.logger import logger


class DataReformatter:
    """
    Reformats VLSP 2025 DRiLL dataset into consistent format for retrieval tasks.
    """
    
    def __init__(self, corpus_path: str, queries_path: str):
        """
        Initialize the data reformatter.
        
        Args:
            corpus_path: Path to legal_corpus.json
            queries_path: Path to train.json
        """
        self.corpus_path = Path(corpus_path)
        self.queries_path = Path(queries_path)
        
    def load_corpus(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Load and reformat the legal corpus.
        
        Returns:
            List of dicts with format:
            {
                "id": str,      # Chunk ID
                "title": str,   # Chunk title (law_id)
                "content": str  # Chunk content (concatenated articles)
            }
        """
        logger.info(f"Loading corpus from {self.corpus_path}")
        
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            raw_corpus = json.load(f)
        
        reformatted_corpus = []
        error_chunks = []
        for chunk in raw_corpus:
            title = chunk.get('law_id', '')
            for idx, article in enumerate(chunk.get('content', [])):
                article_content = article.get('content_Article', '').strip()
                article_id = article.get('aid', f"{title}-{idx}")
                if article_content:
                    reformatted_doc = {
                        "id": article_id,
                        "title": str(title),
                        "content": article_content
                    }
                    reformatted_corpus.append(reformatted_doc)
                else:
                    logger.warning(f"Article {title}-{article_id} has no content")
                    error_chunks.append(article_id)
        
        logger.info(f"Reformatted {len(reformatted_corpus)} chunks")
        return reformatted_corpus, error_chunks
    
    def load_queries(self) -> List[Dict[str, Any]]:
        """
        Load and reformat the training queries.
        
        Returns:
            List of dicts with format:
            {
                "id": str,           # Query ID
                "question": str,     # Question text
                "relevants": List[str]  # List of relevant document IDs
            }
        """
        logger.info(f"Loading queries from {self.queries_path}")
        
        with open(self.queries_path, 'r', encoding='utf-8') as f:
            raw_queries = json.load(f)
        
        reformatted_queries = []
        
        for query in raw_queries:
            # Extract query ID
            query_id = str(query.get('qid', ''))
            # Extract question text
            question = query.get('question', '').strip()
            # Extract relevant document IDs
            relevant_ids = [str(rid) for rid in query.get('relevant_laws', [])]
            
            reformatted_query = {
                "id": query_id,
                "question": question,
                "relevants": relevant_ids
            }
            
            reformatted_queries.append(reformatted_query)
        
        logger.info(f"Reformatted {len(reformatted_queries)} queries")
        return reformatted_queries
    
    def save_reformatted_data(self, corpus_output_path: str, queries_output_path: str):
        """
        Save reformatted data to JSON files.
        
        Args:
            corpus_output_path: Path to save reformatted corpus
            queries_output_path: Path to save reformatted queries
            use_chunk_mapping: Whether to map article IDs to chunk IDs
        """
        # Save corpus
        corpus, error_chunks = self.load_corpus()
        with open(corpus_output_path, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)

        queries = self.load_queries()
        for query in queries:
            for relevant in query['relevants']:
                if relevant in error_chunks:
                    logger.warning(f"Query {query['id']} has relevant {relevant} which is not in corpus")
                    query['relevants'].remove(relevant)
        
        with open(queries_output_path, 'w', encoding='utf-8') as f:
            json.dump(queries, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved reformatted data:")
        logger.info(f"  Corpus: {corpus_output_path} ({len(corpus)} chunks)")
        logger.info(f"  Queries: {queries_output_path} ({len(queries)} queries)")
        
        return corpus, queries