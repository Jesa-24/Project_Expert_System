"""
cli.py - RAG Expert System (Gemini Version)
============================================
Command Line Interface untuk RAG dengan Google Gemini.

Penggunaan:
    python cli.py --index
    python cli.py --chat
    python cli.py --ask "Apa itu machine learning?"
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from rag.document_loader import DocumentLoader
from rag.text_splitter import TextChunker
from rag.vector_store import VectorStoreManager
from rag.rag_chain import RAGChain
from utils.helpers import check_gemini_api_key, normalize_gemini_model, GEMINI_API_KEY_GUIDE

DOCS_DIR = "./data/documents"
VECTORSTORE_DIR = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEMINI_MODEL = normalize_gemini_model(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K_RESULTS", "5"))


def check_prerequisites():
    print("\n" + "="*60)
    print(" RAG EXPERT SYSTEM (Gemini) - Pengecekan Sistem")
    print("="*60)

    print("\n[1/2] Mengecek Gemini API Key...")
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "isi_api_key_kamu_di_sini":
        print("API key belum diset!")
        print(GEMINI_API_KEY_GUIDE)
        return False
    if check_gemini_api_key(GOOGLE_API_KEY):
        print(f"API Key valid, model: {GEMINI_MODEL}")
    else:
        print("API Key tidak valid atau koneksi gagal!")
        return False

    print("\n[2/2] Mengecek folder dokumen...")
    docs_path = Path(DOCS_DIR)
    docs_path.mkdir(parents=True, exist_ok=True)
    supported_ext = {".pdf", ".pptx", ".ppt", ".docx", ".txt"}
    doc_files = [f for f in docs_path.rglob("*")
                 if f.is_file() and f.suffix.lower() in supported_ext]
    if not doc_files:
        print(f"Folder '{DOCS_DIR}' kosong! Letakkan file dokumen di sana.")
    else:
        print(f"Ditemukan {len(doc_files)} file dokumen")

    print("\n" + "="*60 + "\n")
    return True


def index_documents():
    loader = DocumentLoader()
    chunker = TextChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    vs_manager = VectorStoreManager(
        embedding_model=EMBEDDING_MODEL,
        persist_directory=VECTORSTORE_DIR,
    )
    docs = loader.load_directory(DOCS_DIR)
    if not docs:
        print("Tidak ada dokumen yang bisa dimuat!")
        return None
    chunks = chunker.split_documents(docs)
    vs_manager.create_vectorstore(chunks)
    print("Indexing selesai!\n")
    return vs_manager


def load_vectorstore():
    vs_manager = VectorStoreManager(
        embedding_model=EMBEDDING_MODEL,
        persist_directory=VECTORSTORE_DIR,
    )
    result = vs_manager.load_vectorstore()
    if result is None:
        print("Vector store belum ada! Jalankan: python cli.py --index")
        return None
    return vs_manager


def interactive_chat(vs_manager):
    print("\n" + "="*60)
    print(" RAG EXPERT SYSTEM - Chat (Gemini)")
    print("="*60)
    print(f"Model: {GEMINI_MODEL}")
    print("Ketik 'quit' untuk keluar\n")

    rag = RAGChain(
        gemini_model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        top_k_results=TOP_K,
    )
    rag.setup_chain(vs_manager.as_retriever(k=TOP_K))
    print("Sistem siap! Silakan bertanya.\n")

    while True:
        try:
            question = input("Pertanyaan: ").strip()
            if not question:
                continue
            if question.lower() in ("quit", "exit", "keluar"):
                print("Sampai jumpa!")
                break

            print("\nMemproses dengan Gemini...\n")
            print("-" * 60)
            for chunk in rag.stream_answer(question):
                print(chunk, end="", flush=True)
            print("\n" + "-" * 60)

            sources = rag.get_relevant_docs(question)
            print(f"\nSumber ({len(sources)} dokumen):")
            for i, doc in enumerate(sources, 1):
                print(f"  [{i}] {doc.metadata.get('source_file','?')} | Hal. {doc.metadata.get('page','N/A')}")
            print()

        except KeyboardInterrupt:
            print("\nProgram dihentikan.")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def ask_single(question, vs_manager):
    rag = RAGChain(gemini_model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY, top_k_results=TOP_K)
    rag.setup_chain(vs_manager.as_retriever(k=TOP_K))
    result = rag.ask_with_sources(question)
    print(f"\nPertanyaan: {question}")
    print("-" * 60)
    print(result["answer"])
    print(f"\nSumber: {result['num_sources']} dokumen")
    for i, s in enumerate(result["sources"], 1):
        print(f"  [{i}] {s['file']} | Hal. {s['page']}")


def main():
    parser = argparse.ArgumentParser(description="RAG Expert System CLI (Gemini)")
    parser.add_argument("--index", action="store_true", help="Index dokumen")
    parser.add_argument("--chat", action="store_true", help="Mode chat interaktif")
    parser.add_argument("--ask", type=str, help="Tanya satu pertanyaan")
    parser.add_argument("--status", action="store_true", help="Cek status")
    args = parser.parse_args()

    if not any([args.index, args.chat, args.ask, args.status]):
        parser.print_help()
        sys.exit(0)

    if args.status:
        check_prerequisites()
        sys.exit(0)

    if not check_prerequisites():
        sys.exit(1)

    if args.index:
        index_documents()
        sys.exit(0)

    vs_manager = load_vectorstore()
    if vs_manager is None:
        sys.exit(1)

    if args.chat:
        interactive_chat(vs_manager)
    elif args.ask:
        ask_single(args.ask, vs_manager)


if __name__ == "__main__":
    main()
