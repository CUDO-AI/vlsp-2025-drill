#!/usr/bin/env python3
"""
Example usage of FAISS HNSW index for document indexing and retrieval
"""

import numpy as np
import json
import os
from typing import List, Dict, Any
from faiss import FaissStorage, FaissRetriever
from core.logger import logger


def create_sample_documents() -> List[Dict[str, Any]]:
    """Create sample documents for demonstration"""
    documents = [
        {
            "title": "Machine Learning Basics",
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed.",
            "source": "AI Textbook",
            "timestamp": "2024-01-01"
        },
        {
            "title": "Deep Learning Fundamentals",
            "content": "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
            "source": "DL Guide",
            "timestamp": "2024-01-02"
        },
        {
            "title": "Natural Language Processing",
            "content": "NLP is a field of AI that focuses on the interaction between computers and human language.",
            "source": "NLP Handbook",
            "timestamp": "2024-01-03"
        },
        {
            "title": "Computer Vision Applications",
            "content": "Computer vision enables machines to interpret and understand visual information from the world.",
            "source": "CV Journal",
            "timestamp": "2024-01-04"
        },
        {
            "title": "Reinforcement Learning",
            "content": "Reinforcement learning is a type of machine learning where agents learn to make decisions by taking actions in an environment.",
            "source": "RL Papers",
            "timestamp": "2024-01-05"
        }
    ]
    return documents


def create_sample_embeddings(documents: List[Dict[str, Any]], dimension: int = 1024) -> np.ndarray:
    """Create sample embeddings (random for demonstration)"""
    # In practice, you would use a real embedding model
    np.random.seed(42)  # For reproducible results
    embeddings = np.random.randn(len(documents), dimension).astype('float32')
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    return embeddings


def demo_indexing_and_search():
    """Demonstrate FAISS HNSW indexing and search"""
    logger.info("Starting FAISS HNSW demo...")
    
    # Create sample data
    documents = create_sample_documents()
    embeddings = create_sample_embeddings(documents, dimension=1024)
    
    logger.info(f"Created {len(documents)} sample documents with {embeddings.shape[1]}-dimensional embeddings")
    
    # Initialize FAISS storage
    faiss_storage = FaissStorage(
        dimension=1024,
        m=32,  # Number of connections per layer
        ef_construction=200  # Construction parameter
    )
    
    # Add documents to index
    faiss_storage.add_documents(documents, embeddings)
    
    # Create retriever
    retriever = FaissRetriever(faiss_storage)
    
    # Create a sample query embedding
    query_embedding = np.random.randn(1024).astype('float32')
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    # Perform search
    logger.info("Performing search...")
    results = retriever.retrieve(query_embedding, k=3)
    
    # Display results
    print("\n=== Search Results ===")
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Score: {result['score']:.4f}")
        print(f"  Distance: {result['distance']:.4f}")
        print(f"  Title: {result['metadata']['title']}")
        print(f"  Content: {result['metadata']['content'][:100]}...")
        print(f"  Source: {result['metadata']['source']}")
    
    # Get index statistics
    stats = faiss_storage.get_index_stats()
    print(f"\n=== Index Statistics ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return faiss_storage, retriever


def demo_save_and_load():
    """Demonstrate saving and loading FAISS index"""
    logger.info("Demonstrating save and load functionality...")
    
    # Create and populate index
    faiss_storage, retriever = demo_indexing_and_search()
    
    # Save index
    index_path = "faiss_storage/sample_index.faiss"
    documents_path = "faiss_storage/sample_documents.pkl"
    
    # Create directory if it doesn't exist
    os.makedirs("faiss_storage", exist_ok=True)
    
    faiss_storage.save_index(index_path, documents_path)
    
    # Create new storage instance and load index
    new_faiss_storage = FaissStorage()
    new_faiss_storage.load_index(index_path, documents_path)
    
    # Create new retriever
    new_retriever = FaissRetriever(new_faiss_storage)
    
    # Test search with loaded index
    query_embedding = np.random.randn(1024).astype('float32')
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    results = new_retriever.retrieve(query_embedding, k=2)
    
    print(f"\n=== Search Results (Loaded Index) ===")
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Score: {result['score']:.4f}")
        print(f"  Title: {result['metadata']['title']}")
    
    return new_faiss_storage, new_retriever


def demo_batch_retrieval():
    """Demonstrate batch retrieval"""
    logger.info("Demonstrating batch retrieval...")
    
    # Create sample data
    documents = create_sample_documents()
    embeddings = create_sample_embeddings(documents, dimension=1024)
    
    # Initialize storage and add documents
    faiss_storage = FaissStorage(dimension=1024)
    faiss_storage.add_documents(documents, embeddings)
    
    retriever = FaissRetriever(faiss_storage)
    
    # Create multiple query embeddings
    query_embeddings = np.random.randn(3, 1024).astype('float32')
    for i in range(query_embeddings.shape[0]):
        query_embeddings[i] = query_embeddings[i] / np.linalg.norm(query_embeddings[i])
    
    # Perform batch retrieval
    batch_results = retriever.batch_retrieve(query_embeddings, k=2)
    
    print(f"\n=== Batch Retrieval Results ===")
    for i, results in enumerate(batch_results):
        print(f"\nQuery {i+1}:")
        for j, result in enumerate(results):
            print(f"  Result {j+1}: {result['metadata']['title']} (Score: {result['score']:.4f})")


def demo_content_filtering():
    """Demonstrate content filtering"""
    logger.info("Demonstrating content filtering...")
    
    # Create sample data
    documents = create_sample_documents()
    embeddings = create_sample_embeddings(documents, dimension=1024)
    
    # Initialize storage and add documents
    faiss_storage = FaissStorage(dimension=1024)
    faiss_storage.add_documents(documents, embeddings)
    
    retriever = FaissRetriever(faiss_storage)
    
    # Create query embedding
    query_embedding = np.random.randn(1024).astype('float32')
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    # Search with content filter
    results = retriever.search_by_content(
        query_embedding, 
        k=3, 
        content_filter="learning"
    )
    
    print(f"\n=== Content Filtered Results (filter: 'learning') ===")
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Score: {result['score']:.4f}")
        print(f"  Title: {result['metadata']['title']}")
        print(f"  Content: {result['metadata']['content'][:80]}...")


if __name__ == "__main__":
    print("FAISS HNSW Index Demo")
    print("=" * 50)
    
    try:
        # Run all demos
        demo_indexing_and_search()
        print("\n" + "=" * 50)
        
        demo_save_and_load()
        print("\n" + "=" * 50)
        
        demo_batch_retrieval()
        print("\n" + "=" * 50)
        
        demo_content_filtering()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in demo: {e}")
        print(f"Error: {e}") 