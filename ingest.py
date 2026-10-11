import sys
import os
import re
import pickle
import shutil
from pathlib import Path
from typing import List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from config import (
    DATA_DIR, CHROMA_DIR, BM25_INDEX_PATH,
    OPENAI_API_KEY, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, CROPS
)


def load_all_pdfs() -> list:
    all_pages = []

    print("\n📂 [Step 1/3] Scanning and Loading Research PDFs...")
    for crop in CROPS:
        crop_folder = DATA_DIR / crop
        if not crop_folder.exists():
            print(f"⚠️ Folder not found: {crop_folder}")
            continue

        pdf_files = list(crop_folder.glob("*.pdf"))
        print(f"🌿 Crop '{crop}': found {len(pdf_files)} PDF(s)")

        for pdf_path in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_path))
                pages = loader.load()

                for p in pages:
                    p.metadata["crop"] = crop
                    p.metadata["source"] = pdf_path.name
                    p.metadata["page"] = p.metadata.get("page", 0) + 1

                all_pages.extend(pages)
                print(f"   ✅ Loaded '{pdf_path.name}' ({len(pages)} pages)")
            except Exception as e:
                print(f"   ❌ Failed to load '{pdf_path.name}': {e}")

    return all_pages


def split_documents(docs: list) -> list:
    print(f"\n✂️ [Step 2/3] Chunking {len(docs)} pages (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "]
    )
    splits = splitter.split_documents(docs)
    print(f"   ✅ Generated {len(splits)} total semantic chunks.")
    return splits


def build_chroma_store(chunks: list):
    print("\n🔮 [Step 3A/3] Generating Embeddings & Storing in ChromaDB...")

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is missing in your .env file!")

    if CHROMA_DIR.exists():
        print(f"   🧹 Clearing existing ChromaDB at: {CHROMA_DIR}")
        shutil.rmtree(CHROMA_DIR)

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    print(f"   ✅ ChromaDB successfully indexed {len(chunks)} vectors.")
    return vectorstore


def build_bm25_index(chunks: list):
    print("\n⚡ [Step 3B/3] Building BM25 Sparse Keyword Index...")

    def tokenize(text: str) -> List[str]:
        return [w.lower() for w in re.findall(r'\b[a-zA-Z0-9_\-²³]+\b', text)]

    corpus_tokens = [tokenize(chunk.page_content) for chunk in chunks]
    bm25 = BM25Okapi(corpus_tokens)

    bm25_data = {
        "bm25": bm25,
        "chunks": [chunk.page_content for chunk in chunks],
        "metadatas": [chunk.metadata for chunk in chunks]
    }

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_data, f)

    print(f"   ✅ BM25 Index serialized at: {BM25_INDEX_PATH}")


def main():
    print("=" * 65)
    print("🌾 Starting AgriRAG Document Ingestion & Dual Indexing")
    print("=" * 65)

    docs = load_all_pdfs()
    if not docs:
        print("\n❌ No PDF documents found. Please check your data/ folders.")
        return

    chunks = split_documents(docs)
    build_chroma_store(chunks)
    build_bm25_index(chunks)

    print("\n" + "=" * 65)
    print("🎉 Ingestion Pipeline Finished Successfully!")
    print("   • Dense Vector Store: storage/chroma_db/")
    print("   • Sparse BM25 Index:  storage/bm25_index.pkl")
    print("=" * 65)


if __name__ == "__main__":
    main()
