"""
Retrieval Data Classes and Utilities

This module provides data classes and utilities for working with
standardized retrieval data in the VLSP 2025 DRiLL task.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json


@dataclass
class Chunk:
    """Standardized chunk format for retrieval."""
    id: str
    title: str
    content: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk':
        """Create Chunk from dictionary."""
        return cls(
            id=str(data.get('id', '')),
            title=str(data.get('title', '')),
            content=str(data.get('content', ''))
        )


@dataclass
class Query:
    """Standardized query format for retrieval."""
    id: str
    question: str
    relevants: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "question": self.question,
            "relevants": self.relevants
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Query':
        """Create Query from dictionary."""
        return cls(
            id=str(data.get('id', '')),
            question=str(data.get('question', '')),
            relevants=[str(rid) for rid in data.get('relevants', [])]
        )


@dataclass
class RetrievalDataset:
    """Container for standardized retrieval dataset."""
    corpus: List[Chunk]
    queries: List[Query]
    
    def __len__(self) -> int:
        """Return number of queries."""
        return len(self.queries)
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Get chunk by ID."""
        for chunk in self.corpus:
            if chunk.id == chunk_id:
                return chunk
        return None
    
    def get_query_by_id(self, query_id: str) -> Optional[Query]:
        """Get query by ID."""
        for query in self.queries:
            if query.id == query_id:
                return query
        return None
    
    def get_relevant_chunks(self, query_id: str) -> List[Chunk]:
        """Get relevant chunks for a query."""
        query = self.get_query_by_id(query_id)
        if not query:
            return []
        
        relevant_chunks = []
        for chunk_id in query.relevants:
            chunk = self.get_chunk_by_id(chunk_id)
            if chunk:
                relevant_chunks.append(chunk)
        
        return relevant_chunks
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "corpus": [doc.to_dict() for doc in self.corpus],
            "queries": [query.to_dict() for query in self.queries]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetrievalDataset':
        """Create RetrievalDataset from dictionary."""
        corpus = [Chunk.from_dict(chunk_data) for chunk_data in data.get('corpus', [])]
        queries = [Query.from_dict(query_data) for query_data in data.get('queries', [])]
        return cls(corpus=corpus, queries=queries)
    
    def save(self, filepath: str) -> None:
        """Save dataset to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'RetrievalDataset':
        """Load dataset from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def save_separate(self, corpus_path: str, queries_path: str) -> None:
        """Save corpus and queries to separate files."""
        # Save corpus
        corpus_data = [doc.to_dict() for doc in self.corpus]
        with open(corpus_path, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, ensure_ascii=False, indent=2)
        
        # Save queries
        queries_data = [query.to_dict() for query in self.queries]
        with open(queries_path, 'w', encoding='utf-8') as f:
            json.dump(queries_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_separate(cls, corpus_path: str, queries_path: str) -> 'RetrievalDataset':
        """Load dataset from separate corpus and queries files."""
        # Load corpus
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
        corpus = [Chunk.from_dict(chunk_data) for chunk_data in corpus_data]
        
        # Load queries
        with open(queries_path, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
        queries = [Query.from_dict(query_data) for query_data in queries_data]
        
        return cls(corpus=corpus, queries=queries)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        total_chunks = len(self.corpus)
        total_queries = len(self.queries)
        
        # Calculate average content length
        avg_content_length = sum(len(chunk.content) for chunk in self.corpus) / total_chunks if total_chunks > 0 else 0
        
        # Calculate average question length
        avg_question_length = sum(len(query.question) for query in self.queries) / total_queries if total_queries > 0 else 0
        
        # Calculate average number of relevant documents per query
        avg_relevants = sum(len(query.relevants) for query in self.queries) / total_queries if total_queries > 0 else 0
        
        # Get unique relevant document IDs
        all_relevant_ids = set()
        for query in self.queries:
            all_relevant_ids.update(query.relevants)
        
        return {
            "total_chunks": total_chunks,
            "total_queries": total_queries,
            "average_content_length": round(avg_content_length, 2),
            "average_question_length": round(avg_question_length, 2),
            "average_relevant_chunks_per_query": round(avg_relevants, 2),
            "unique_relevant_chunks": len(all_relevant_ids),
            "coverage_ratio": round(len(all_relevant_ids) / total_chunks, 4) if total_chunks > 0 else 0
        }