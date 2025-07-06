#!/usr/bin/env python3
"""
Main script for FAISS HNSW indexing and retrieval
"""

import argparse
import numpy as np
import os
from typing import List, Dict, Any
from faiss import FaissIndexer, FaissRetriever
from core.logger import logger


def main():
    parser = argparse.ArgumentParser(description="FAISS HNSW Indexing and Retrieval")
    parser.add_argument("--mode", choices=["index", "search", "demo"], 
                       default="demo", help="Mode to run")
    parser.add_argument("--corpus", type=str, help="Path to corpus JSON file")
    parser.add_argument("--embeddings", type=str, help="Path to embeddings numpy file")
    parser.add_argument("--index-dir", type=str, default="faiss_storage", 
                       help="Directory to save/load index")
    parser.add_argument("--index-name", type=str, default="faiss_index", 
                       help="Name of the index")
    parser.add_argument("--dimension", type=int, default=1024, 
                       help="Dimension of embeddings")
    parser.add_argument("--m", type=int, default=32, 
                       help="Number of connections per layer in HNSW")
    parser.add_argument("--ef-construction", type=int, default=200, 
                       help="Construction parameter for HNSW")
    parser.add_argument("--overwrite", action="store_true", 
                       help="Overwrite existing index")
    parser.add_argument("--k", type=int, default=10, 
                       help="Number of results to return")
    parser.add_argument("--query-embedding", type=str, 
                       help="Path to query embedding numpy file")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        run_demo()
    elif args.mode == "index":
        run_indexing(args)
    elif args.mode == "search":
        run_search(args)


def run_demo():
    """Run demonstration"""
    print("Running FAISS HNSW Demo...")
    
    # Import and run demo
    from faiss.example_usage import (
        demo_indexing_and_search,
        demo_save_and_load,
        demo_batch_retrieval,
        demo_content_filtering
    )
    
    try:
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


def run_indexing(args):
    """Run indexing mode"""
    if not args.corpus or not args.embeddings:
        print("Error: --corpus and --embeddings are required for indexing mode")
        return
        
    if not os.path.exists(args.corpus):
        print(f"Error: Corpus file {args.corpus} not found")
        return
        
    if not os.path.exists(args.embeddings):
        print(f"Error: Embeddings file {args.embeddings} not found")
        return
    
    print(f"Starting FAISS indexing...")
    print(f"Corpus: {args.corpus}")
    print(f"Embeddings: {args.embeddings}")
    print(f"Index directory: {args.index_dir}")
    print(f"Index name: {args.index_name}")
    print(f"Dimension: {args.dimension}")
    print(f"HNSW m: {args.m}")
    print(f"HNSW ef_construction: {args.ef_construction}")
    
    try:
        # Initialize indexer
        indexer = FaissIndexer(
            dimension=args.dimension,
            m=args.m,
            ef_construction=args.ef_construction
        )
        
        # Run indexing
        index_path = indexer.indexing(
            corpus_path=args.corpus,
            embedding_path=args.embeddings,
            index_dir=args.index_dir,
            index_name=args.index_name,
            overwrite=args.overwrite
        )
        
        print(f"Indexing completed successfully!")
        print(f"Index saved to: {index_path}")
        
        # Print statistics
        stats = indexer.get_stats()
        print(f"\nIndex Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        print(f"Error: {e}")


def run_search(args):
    """Run search mode"""
    index_path = os.path.join(args.index_dir, f"{args.index_name}.faiss")
    documents_path = os.path.join(args.index_dir, f"{args.index_name}_documents.pkl")
    
    if not os.path.exists(index_path):
        print(f"Error: Index file {index_path} not found. Run indexing first.")
        return
        
    if not os.path.exists(documents_path):
        print(f"Error: Documents file {documents_path} not found. Run indexing first.")
        return
    
    print(f"Starting FAISS search...")
    print(f"Index: {index_path}")
    print(f"Documents: {documents_path}")
    print(f"k: {args.k}")
    
    try:
        # Initialize indexer and load index
        indexer = FaissIndexer(dimension=args.dimension)
        indexer.load_index(index_path, documents_path)
        
        # Create retriever
        retriever = FaissRetriever(indexer.storage)
        
        if args.query_embedding and os.path.exists(args.query_embedding):
            # Load query embedding from file
            query_embedding = np.load(args.query_embedding)
            print(f"Loaded query embedding from: {args.query_embedding}")
        else:
            # Create random query embedding for demonstration
            query_embedding = np.random.randn(args.dimension).astype('float32')
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            print("Using random query embedding for demonstration")
        
        # Perform search
        results = retriever.retrieve(query_embedding, k=args.k)
        
        # Display results
        print(f"\n=== Search Results (k={args.k}) ===")
        for i, result in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"  Score: {result['score']:.4f}")
            print(f"  Distance: {result['distance']:.4f}")
            print(f"  ID: {result['id']}")
            if result['metadata']:
                print(f"  Title: {result['metadata']['title']}")
                print(f"  Content: {result['metadata']['content'][:100]}...")
                print(f"  Source: {result['metadata']['source']}")
        
        # Print statistics
        stats = indexer.get_stats()
        print(f"\nIndex Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        logger.error(f"Error during search: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 