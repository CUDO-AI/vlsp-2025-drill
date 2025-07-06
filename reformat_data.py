from core.datasets.data_reformatter import DataReformatter


def main():
    # Initialize reformatter
    reformatter = DataReformatter(
        corpus_path="ir-datasets/original/legal_corpus.json",
        queries_path="ir-datasets/original/train.json"
    )
    
    # Save reformatted data
    reformatter.save_reformatted_data(
        corpus_output_path="ir-datasets/reformatted/corpus.json",
        queries_output_path="ir-datasets/reformatted/queries.json",
        use_chunk_mapping=True
    )
    
    # Example: Load and inspect data
    corpus = reformatter.load_corpus()
    queries = reformatter.reformat_queries_with_mapping()
    
    print(f"\nCorpus example (first chunk):")
    print(f"ID: {corpus[0]['id']}")
    print(f"Title: {corpus[0]['title']}")
    print(f"Content length: {len(corpus[0]['content'])} characters")
    
    print(f"\nQueries example (first query):")
    print(f"ID: {queries[0]['id']}")
    print(f"Question: {queries[0]['question']}")
    print(f"Relevant chunks: {queries[0]['relevants']}")


if __name__ == "__main__":
    main() 