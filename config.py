import os
from pathlib import Path
from dotenv import load_dotenv

# 1. Base Paths
BASE_DIR = Path(__file__).parent.resolve()
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma_db"
BM25_INDEX_PATH = STORAGE_DIR / "bm25_index.pkl"

# Ensure storage directory exists
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# 2. Model Configurations
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 1200

# 3. Chunking & Retrieval Parameters
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4
RRF_K = 60

# 4. Crop Taxonomy
CROPS = ["basil","capsicum","cucumbers","lettuce","strawberries","tomatoes"]
