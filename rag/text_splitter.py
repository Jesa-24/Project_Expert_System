"""
rag/text_splitter.py
====================
Modul untuk memecah dokumen menjadi chunk-chunk kecil
agar lebih efisien saat diproses dan dicari.
"""

import re
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    """
    Memecah dokumen besar menjadi chunk-chunk kecil (potongan teks).
    
    Menggunakan RecursiveCharacterTextSplitter yang memecah teks
    secara rekursif berdasarkan separator (paragraf → kalimat → kata).
    
    Juga mendeteksi posisi chunk dalam dokumen (awal, tengah, akhir)
    untuk membantu pencarian bagian khusus seperti daftar pustaka.
    """

    # Pattern untuk mendeteksi bagian akhir dokumen
    END_PATTERNS = [
        r'daftar\s+pustaka',
        r'bibliografi',
        r'referensi',
        r'laman\s+referensi',
        r'cited\s+references',
        r'bibliography',
        r'references\s*$',
    ]

    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 400,
    ):
        """
        Args:
            chunk_size: Ukuran maksimal setiap chunk (dalam karakter)
            chunk_overlap: Jumlah karakter yang overlap antar chunk
                           (membantu menjaga konteks antar chunk)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            # Separator diurutkan dari yang paling diutamakan
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Memecah list dokumen menjadi chunk-chunk kecil.

        Args:
            documents: List dokumen yang sudah dimuat

        Returns:
            List dokumen yang sudah dipecah menjadi chunk
        """
        if not documents:
            print("⚠️  Tidak ada dokumen untuk dipecah.")
            return []

        print(f"[✂️ ] Memecah {len(documents)} dokumen menjadi chunks...")
        print(f"     Ukuran chunk: {self.chunk_size} karakter")
        print(f"     Overlap     : {self.chunk_overlap} karakter")

        chunks = self.splitter.split_documents(documents)

        # Tambahkan nomor chunk dan posisi dalam dokumen ke metadata
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_total"] = total_chunks
            
            # Deteksi posisi dalam dokumen (awal/tengah/akhir)
            position_ratio = i / total_chunks if total_chunks > 0 else 0
            if position_ratio < 0.3:
                chunk.metadata["position"] = "awal"
            elif position_ratio < 0.7:
                chunk.metadata["position"] = "tengah"
            else:
                chunk.metadata["position"] = "akhir"
            
            # Deteksi apakah chunk ini berisi bagian daftar pustaka/referensi
            content_lower = chunk.page_content.lower()
            for pattern in self.END_PATTERNS:
                if re.search(pattern, content_lower):
                    chunk.metadata["has_references"] = True
                    break
            else:
                chunk.metadata["has_references"] = False

        print(f"     ✅ Total chunks dihasilkan: {len(chunks)}\n")
        return chunks

    def split_text(self, text: str) -> List[str]:
        """
        Memecah string teks biasa menjadi list string chunks.

        Args:
            text: String teks yang akan dipecah

        Returns:
            List string chunks
        """
        return self.splitter.split_text(text)

    def get_stats(self, chunks: List[Document]) -> dict:
        """
        Menghitung statistik dari hasil chunking.

        Returns:
            Dictionary berisi statistik chunk
        """
        if not chunks:
            return {}

        lengths = [len(c.page_content) for c in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "total_chars": sum(lengths),
        }
