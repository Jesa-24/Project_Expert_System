"""
rag/rag_chain.py
================
Modul utama RAG Chain menggunakan Google Gemini sebagai LLM.
"""

import os
from typing import Generator, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from utils.helpers import normalize_gemini_model

RAG_PROMPT_TEMPLATE = """Kamu adalah asisten AI yang membantu menjawab pertanyaan berdasarkan dokumen yang diberikan.

INSTRUKSI:
- Jawab HANYA berdasarkan konteks/dokumen yang disediakan di bawah.
- Jika informasi tidak ada di konteks, katakan "Saya tidak menemukan informasi tersebut di dokumen yang tersedia."
- Berikan jawaban yang jelas, terstruktur, dan informatif.
- Jika relevan, sebutkan sumber informasi dari dokumen mana.
- Jawab dalam Bahasa Indonesia kecuali jika pertanyaan dalam Bahasa Inggris.
- PERHATIAN KHUSUS: Jika pertanyaan tentang daftar pustaka, referensi, atau bibliografi, carilah di bagian akhir dokumen. Bagian ini biasanya berisi informasi lengkap tentang sumber-sumber yang digunakan penulis.

KONTEKS DARI DOKUMEN:
{context}

PERTANYAAN: {question}

JAWABAN:"""


def format_docs(docs: List[Document]) -> str:
    formatted_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "Dokumen")
        page = doc.metadata.get("page", "")
        page_info = f" (hal. {page})" if page else ""
        formatted_parts.append(
            f"[Sumber {i}: {source}{page_info}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted_parts)


class RAGChain:
    """RAG Chain menggunakan Google Gemini sebagai LLM."""

    def __init__(
        self,
        gemini_model: str = "gemini-2.5-flash",
        google_api_key: Optional[str] = None,
        temperature: float = 0.1,
        top_k_results: int = 5,
    ):
        self.model_name = normalize_gemini_model(gemini_model)
        self.top_k = top_k_results
        self.retriever = None
        self.chain = None

        api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "isi_api_key_kamu_di_sini":
            raise ValueError(
                "GOOGLE_API_KEY belum diset!\n"
                "Dapatkan API key gratis di: https://aistudio.google.com/app/apikey\n"
                "Lalu isi di file .env: GOOGLE_API_KEY=api_key_kamu"
            )

        print(f"[Inisialisasi] LLM: Google Gemini ({self.model_name})")

        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=api_key,
            temperature=temperature,
        )

        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        self.output_parser = StrOutputParser()
        print("     Gemini siap!\n")

    def setup_chain(self, retriever) -> None:
        self.retriever = retriever
        self.chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | self.output_parser
        )
        print("[OK] RAG chain berhasil dirakit!\n")

    def ask(self, question: str) -> str:
        if self.chain is None:
            raise ValueError("RAG chain belum diinisialisasi!")
        return self.chain.invoke(question)

    def ask_with_fallback(self, question: str) -> str:
        """
        Menjawab pertanyaan dengan menggunakan fallback search.
        Berguna untuk pertanyaan tentang bagian akhir dokumen seperti daftar pustaka.
        """
        if self.chain is None:
            raise ValueError("RAG chain belum diinisialisasi!")
        
        # Cek apakah retriever memiliki method search_with_fallback
        if hasattr(self.retriever, 'vectorstore') and hasattr(self.retriever.vectorstore, 'search_with_fallback'):
            # Gunakan fallback search
            docs = self.retriever.vectorstore.search_with_fallback(question, k=self.top_k)
            if docs:
                from langchain_core.runnables import RunnablePassthrough
                formatted_context = format_docs(docs)
                # Invoke chain dengan context manual
                return self.chain.invoke({"context": formatted_context, "question": question})
        
        # Fallback ke method biasa
        return self.ask(question)

    def ask_with_sources(self, question: str) -> dict:
        if self.chain is None:
            raise ValueError("RAG chain belum diinisialisasi!")

        relevant_docs = self.retriever.invoke(question)
        answer = self.chain.invoke(question)

        sources = []
        seen = set()
        for doc in relevant_docs:
            file = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "N/A")
            key = f"{file}_{page}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "file": file,
                    "file_type": doc.metadata.get("file_type", "Unknown"),
                    "page": page,
                    "preview": (
                        doc.page_content[:200] + "..."
                        if len(doc.page_content) > 200
                        else doc.page_content
                    ),
                })

        return {"answer": answer, "sources": sources, "num_sources": len(sources)}

    def stream_answer(self, question: str) -> Generator[str, None, None]:
        if self.chain is None:
            yield "Error: RAG chain belum diinisialisasi!"
            return
        for chunk in self.chain.stream(question):
            yield chunk

    def get_relevant_docs(self, question: str) -> List[Document]:
        if self.retriever is None:
            raise ValueError("Retriever belum diinisialisasi!")
        return self.retriever.invoke(question)

    def is_ready(self) -> bool:
        return self.chain is not None
