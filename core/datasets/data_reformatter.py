"""
Data Reformatter for VLSP 2025 DRiLL Retrieval Task

This module provides utilities to reformat the original dataset format
into a consistent format for retrieval tasks.
"""

import json
from typing import List, Dict, Any
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
        
    def load_corpus(self) -> List[Dict[str, Any]]:
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
        
        for chunk in raw_corpus:
            # Extract chunk ID
            chunk_id = str(chunk.get('id', ''))
            
            # Extract title (law_id)
            title = chunk.get('law_id', '')
            
            # Extract and concatenate content from all articles
            content_parts = []
            for article in chunk.get('content', []):
                article_content = article.get('content_Article', '').strip()
                if article_content:
                    content_parts.append(article_content)
            
            content = '\n\n'.join(content_parts)
            
            reformatted_doc = {
                "id": chunk_id,
                "title": str(title),
                "content": content
            }
            
            reformatted_corpus.append(reformatted_doc)
        
        logger.info(f"Reformatted {len(reformatted_corpus)} chunks")
        return reformatted_corpus
    
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
    
    def create_article_mapping(self) -> Dict[str, str]:
        """
        Create a mapping from article IDs to chunk IDs.
        
        Returns:
            Dict mapping article ID to chunk ID
        """
        logger.info("Creating article to document mapping")
        
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            raw_corpus = json.load(f)
        
        article_mapping = {}
        
        for chunk in raw_corpus:
            chunk_id = str(chunk.get('id', ''))
            for article in chunk.get('content', []):
                article_id = str(article.get('aid', ''))
                article_mapping[article_id] = chunk_id
        
        logger.info(f"Created mapping for {len(article_mapping)} articles")
        return article_mapping
    
    def reformat_queries_with_mapping(self) -> List[Dict[str, Any]]:
        """
        Load queries and map article IDs to chunk IDs.
        
        Returns:
            List of reformatted queries with chunk-level relevants
        """
        queries = self.load_queries()
        article_mapping = self.create_article_mapping()
        
        reformatted_queries = []
        
        for query in queries:
            relevant_chunks = set()
            for article_id in query['relevants']:
                if article_id in article_mapping:
                    relevant_chunks.add(article_mapping[article_id])
            
            reformatted_query = {
                "id": query['id'],
                "question": query['question'],
                "relevants": list(relevant_chunks)
            }
            
            reformatted_queries.append(reformatted_query)
        
        logger.info(f"Reformatted {len(reformatted_queries)} queries with chunk mapping")
        return reformatted_queries
    
    def save_reformatted_data(self, 
                             corpus_output_path: str,
                             queries_output_path: str,
                             use_chunk_mapping: bool = True) -> None:
        """
        Save reformatted data to JSON files.
        
        Args:
            corpus_output_path: Path to save reformatted corpus
            queries_output_path: Path to save reformatted queries
            use_chunk_mapping: Whether to map article IDs to chunk IDs
        """
        # Save corpus
        corpus = self.load_corpus()
        with open(corpus_output_path, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)
        
        # Save queries
        if use_chunk_mapping:
            queries = self.reformat_queries_with_mapping()
        else:
            queries = self.load_queries()
        
        with open(queries_output_path, 'w', encoding='utf-8') as f:
            json.dump(queries, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved reformatted data:")
        logger.info(f"  Corpus: {corpus_output_path} ({len(corpus)} chunks)")
        logger.info(f"  Queries: {queries_output_path} ({len(queries)} queries)")