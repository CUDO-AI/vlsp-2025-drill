from core.encoders.ncp import NCPEncoder
import os
import json
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def validate_env_vars():
    """Validate that required environment variables are set"""
    required_vars = [
        "NCP_EMBEDDING_BASE_URL",
        "NCP_EMBEDDING_API_KEY", 
        "NCP_EMBEDDING_MODEL_NAME"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease create a .env file with the following variables:")
        print("NCP_EMBEDDING_BASE_URL=https://mkp-api.fptcloud.com")
        print("NCP_EMBEDDING_API_KEY=your_api_key_here")
        print("NCP_EMBEDDING_MODEL_NAME=your_model_name_here")
        return False
    
    print("✅ All required environment variables are set")
    return True


def embed_corpus(corpus_path: str, output_path: str):
    # Validate environment variables first
    if not validate_env_vars():
        return
    
    print(f"🔧 Initializing NCP Encoder...")
    ncp_encoder = NCPEncoder(
        base_url=os.getenv("NCP_EMBEDDING_BASE_URL"),
        api_key=os.getenv("NCP_EMBEDDING_API_KEY"),
        model_name=os.getenv("NCP_EMBEDDING_MODEL_NAME")
    )
    
    print(f"📖 Loading corpus from {corpus_path}")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    
    print(f"📝 Preparing {len(corpus)} passages for embedding...")
    passages = [chunk['title'] + '\n' + chunk['content'] for chunk in corpus]
    
    print(f"🚀 Starting embedding process...")
    embeddings = ncp_encoder.encode_passages(passages[:2], batch_size=2)
    
    print(f"💾 Saving embeddings to {output_path}")
    np.save(output_path, embeddings)
    print(f"✅ Embeddings saved successfully!")


if __name__ == "__main__":
    embed_corpus(
        corpus_path="ir-datasets/reformatted/corpus.json",
        output_path="ir-datasets/embeddings/corpus.npy"
    )