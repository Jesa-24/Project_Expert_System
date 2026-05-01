"""
rag/vector_store.py
===================
Modul untuk menyimpan dan mencari dokumen menggunakan
ChromaDB sebagai vector database dan HuggingFace Embeddings.

Alur:
1. Dokumen di-embed (diubah menjadi vektor numerik)
2. Vektor disimpan di ChromaDB
3. Saat query, pertanyaan juga di-embed
4. Dicari vektor yang paling mirip (cosine similarity)
"""

import os
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Chroma telemetry is optional and currently noisy on some local setups.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


class VectorStoreManager:
    """
    Mengelola penyimpanan dan pencarian vektor dokumen menggunakan ChromaDB.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str = "./vectorstore",
        collection_name: str = "rag_documents",
    ):
        """
        Args:
            embedding_model: Nama model HuggingFace untuk embedding
                             (default: all-MiniLM-L6-v2 - ringan & bagus)
            persist_directory: Folder penyimpanan ChromaDB
            collection_name: Nama koleksi di ChromaDB
        """
        self.embedding_model_name = embedding_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vectorstore: Optional[Chroma] = None

        print(f"[🔧] Inisialisasi Embedding Model: {embedding_model}")
        print(f"     (Download pertama kali mungkin butuh beberapa menit...)\n")

        # Inisialisasi model embedding
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},   # Ganti ke "cuda" jika punya GPU
            encode_kwargs={"normalize_embeddings": True},
        )
        print("     ✅ Embedding model siap!\n")

    def create_vectorstore(self, chunks: List[Document]) -> Chroma:
        """
        Membuat vector store baru dari chunks dokumen.
        Jika sudah ada, data lama akan dihapus dan dibuat ulang.

        Args:
            chunks: List Document chunks yang sudah dipecah

        Returns:
            Chroma vectorstore yang siap digunakan
        """
        if not chunks:
            raise ValueError("Tidak ada chunks untuk dimasukkan ke vector store!")

        print(f"[💾] Membuat vector store dari {len(chunks)} chunks...")
        print(f"     Database: ChromaDB")
        print(f"     Lokasi  : {self.persist_directory}")
        print(f"     Koleksi : {self.collection_name}\n")

        # Hapus vectorstore lama jika ada
        if os.path.exists(self.persist_directory):
            import shutil
            shutil.rmtree(self.persist_directory)
            print("     🗑️  Vector store lama dihapus.")

        # Buat vectorstore baru
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )

        print(f"     ✅ Vector store berhasil dibuat!\n")
        return self.vectorstore

    def load_vectorstore(self) -> Optional[Chroma]:
        """
        Memuat vector store yang sudah ada sebelumnya.

        Returns:
            Chroma vectorstore atau None jika belum ada
        """
        if not os.path.exists(self.persist_directory):
            print("⚠️  Vector store belum ada. Silakan upload dokumen terlebih dahulu.")
            return None

        print(f"[📂] Memuat vector store dari '{self.persist_directory}'...")

        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )

        count = self.vectorstore._collection.count()
        print(f"     ✅ Vector store dimuat: {count} chunks tersimpan\n")
        return self.vectorstore

    def add_documents(self, chunks: List[Document]) -> None:
        """
        Menambahkan dokumen baru ke vector store yang sudah ada.

        Args:
            chunks: List Document chunks baru
        """
        if self.vectorstore is None:
            self.vectorstore = self.load_vectorstore()
            if self.vectorstore is None:
                self.create_vectorstore(chunks)
                return

        print(f"[➕] Menambahkan {len(chunks)} chunks ke vector store...")
        self.vectorstore.add_documents(chunks)
        print(f"     ✅ Berhasil ditambahkan!\n")

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.3,
    ) -> List[Tuple[Document, float]]:
        """
        Mencari dokumen yang paling relevan dengan query.

        Args:
            query: Pertanyaan/query dari pengguna
            k: Jumlah dokumen yang dikembalikan
            score_threshold: Threshold minimum similarity score

        Returns:
            List tuple (Document, score) yang paling relevan
        """
        if self.vectorstore is None:
            raise ValueError("Vector store belum diinisialisasi!")

        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results

    def as_retriever(self, k: int = 5):
        """
        Mengembalikan retriever untuk digunakan di RAG chain.

        Args:
            k: Jumlah dokumen yang dikembalikan per query

        Returns:
            LangChain retriever object
        """
        if self.vectorstore is None:
            raise ValueError("Vector store belum diinisialisasi!")

        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def get_document_count(self) -> int:
        """Mengembalikan jumlah chunks yang tersimpan."""
        if self.vectorstore is None:
            return 0
        return self.vectorstore._collection.count()

    def is_ready(self) -> bool:
        """Mengecek apakah vector store sudah siap digunakan."""
        return self.vectorstore is not None and self.get_document_count() > 0

    def search_with_fallback(
        self,
        query: str,
        k: int = 5,
    ) -> List[Document]:
        """
        Mencari dokumen dengan fallback untuk bagian akhir dokumen.
        
        Jika pencarian pertama tidak menemukan hasil yang baik,
        akan melakukan pencarian tambahan dengan kata kunci khusus
        untuk bagian daftar pustaka/referensi.

        Args:
            query: Pertanyaan dari pengguna
            k: Jumlah dokumen yang dikembalikan

        Returns:
            List dokumen yang relevan
        """
        if self.vectorstore is None:
            raise ValueError("Vector store belum diinisialisasi!")
        
        # Pertama, coba pencarian biasa
        results = self.vectorstore.similarity_search(query, k=k)
        
        # Cek apakah ada hasil yang berisi referensi
        has_reference_results = any(
            doc.metadata.get("has_references", False) 
            for doc in results
        )
        
        if not has_reference_results:
            # Fallback: cari dengan kata kunci khusus
            fallback_keywords = [
                "referensi", "daftar pustaka", "bibliografi",
                "references", "bibliography", "cited"
            ]
            
            for keyword in fallback_keywords:
                try:
                    fallback_results = self.vectorstore.similarity_search(
                        keyword, k=k
                    )
                    # Gabungkan dengan hasil yang ada
                    for doc in fallback_results:
                        if doc not in results:
                            results.append(doc)
                except Exception:
                    continue
        
        return results[:k]
