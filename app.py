"""
app.py - RAG Expert System (Gemini Version)
===========================================
Streamlit UI untuk RAG menggunakan Google Gemini API.

Jalankan: streamlit run app.py
"""

import os
import time
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from rag.document_loader import DocumentLoader
from rag.text_splitter import TextChunker
from rag.vector_store import VectorStoreManager
from rag.rag_chain import RAGChain
from utils.helpers import (
    check_gemini_api_key,
    get_gemini_models,
    normalize_gemini_model,
    save_uploaded_file,
    get_documents_info,
    GEMINI_API_KEY_GUIDE,
)

# ── Konfigurasi ──
DOCS_DIR = "./data/documents"
VECTORSTORE_DIR = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_MODEL = normalize_gemini_model(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K_RESULTS", "5"))

Path(DOCS_DIR).mkdir(parents=True, exist_ok=True)
Path(VECTORSTORE_DIR).mkdir(parents=True, exist_ok=True)

# ── Page Config ──
st.set_page_config(
    page_title="RAG Expert System - Gemini",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size:2rem; font-weight:bold; color:#4285f4;
                   text-align:center; padding:1rem 0; }
    .status-ok   { background:#d4edda; border:1px solid #c3e6cb;
                   color:#155724; padding:.5rem 1rem; border-radius:.5rem; margin:.3rem 0; }
    .status-err  { background:#f8d7da; border:1px solid #f5c6cb;
                   color:#721c24; padding:.5rem 1rem; border-radius:.5rem; margin:.3rem 0; }
    .source-card { background:#f8f9fa; border-left:4px solid #4285f4;
                   padding:.5rem 1rem; margin:.4rem 0;
                   border-radius:0 .25rem .25rem 0; font-size:.85rem; }
    .gemini-badge { background:linear-gradient(135deg,#4285f4,#34a853,#fbbc04,#ea4335);
                    color:white; padding:.2rem .8rem; border-radius:1rem;
                    font-size:.8rem; font-weight:bold; display:inline-block; }
</style>
""", unsafe_allow_html=True)


# ── Session State ──
def init_state():
    defaults = {
        "messages": [],
        "rag_chain": None,
        "vector_store": None,
        "documents_indexed": False,
        "selected_model": DEFAULT_MODEL,
        "show_sources": True,
        "api_key": os.getenv("GOOGLE_API_KEY", ""),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


@st.cache_resource
def get_vs_manager():
    return VectorStoreManager(
        embedding_model=EMBEDDING_MODEL,
        persist_directory=VECTORSTORE_DIR,
    )


def setup_rag_chain(vs_manager):
    rag = RAGChain(
        gemini_model=st.session_state.selected_model,
        google_api_key=st.session_state.api_key,
        temperature=0.1,
        top_k_results=TOP_K,
    )
    retriever = vs_manager.as_retriever(k=TOP_K)
    rag.setup_chain(retriever)
    st.session_state.rag_chain = rag


def process_documents():
    loader = DocumentLoader()
    chunker = TextChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    vs_manager = get_vs_manager()

    with st.spinner("📄 Memuat dokumen..."):
        docs = loader.load_directory(DOCS_DIR)
    if not docs:
        st.error("❌Tidak ada dokumen yang berhasil dimuat!")
        return None

    with st.spinner("✂️ Memecah dokumen menjadi chunks..."):
        chunks = chunker.split_documents(docs)

    with st.spinner(f"💾Mengindeks {len(chunks)} chunks ke ChromaDB..."):
        vs_manager.create_vectorstore(chunks)

    st.session_state.vector_store = vs_manager
    setup_rag_chain(vs_manager)
    st.session_state.documents_indexed = True
    return chunker.get_stats(chunks)


def load_existing():
    vs_manager = get_vs_manager()
    result = vs_manager.load_vectorstore()
    if result:
        st.session_state.vector_store = vs_manager
        setup_rag_chain(vs_manager)
        st.session_state.documents_indexed = True
        return True
    return False


# ════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Pengaturan")
    # st.markdown('<span class="gemini-badge">Powered by Gemini</span>', unsafe_allow_html=True)
    st.divider()

    # ── API Key ──
    st.markdown("###  Google Gemini API Key")
    api_key_input = st.text_input(
        "API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="AIza...",
        help="Ambil di aistudio.google.com"
    )

    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        # Reset chain agar rebuild dengan key baru
        st.session_state.rag_chain = None

    # Validasi API key
    if st.session_state.api_key and st.session_state.api_key != "isi_api_key_kamu_di_sini":
        key_valid = check_gemini_api_key(st.session_state.api_key)
        if key_valid:
            st.markdown('<div class="status-ok">API Key: Valid</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-err">API Key: Tidak valid</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-err">API Key belum diisi</div>', unsafe_allow_html=True)
        with st.expander("Cara Dapat API Key (Gratis)"):
            st.markdown("""
1. Buka [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Login dengan akun Google
3. Klik **"Create API Key"**
4. Salin & paste di sini

Gunakan model yang masih aktif seperti `gemini-2.5-flash`.
            """)

    st.divider()

    # ── Pilih Model ──
    st.markdown("###  Model Gemini")
    models = get_gemini_models()
    model_names = list(models.keys())
    selected_idx = model_names.index(DEFAULT_MODEL) if DEFAULT_MODEL in model_names else 0

    selected_model = st.selectbox(
        "Pilih Model",
        options=model_names,
        index=selected_idx,
        format_func=lambda m: f"{m}  —  {models[m]}",
        help="gemini-2.5-flash direkomendasikan untuk RAG"
    )
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        if st.session_state.vector_store:
            setup_rag_chain(st.session_state.vector_store)

    st.divider()

    # ── Upload Dokumen ──
    st.markdown("### Upload Dokumen")
    uploaded_files = st.file_uploader(
        "PDF, PPTX, DOCX, TXT",
        type=["pdf", "pptx", "ppt", "docx", "txt"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        for uf in uploaded_files:
            save_uploaded_file(uf, DOCS_DIR)
        st.success(f" {len(uploaded_files)} file disimpan!")

    # ── Daftar Dokumen ──
    st.markdown("### Dokumen Tersimpan")
    docs_info = get_documents_info(DOCS_DIR)
    if docs_info:
        for d in docs_info:
            st.markdown(f" **{d['name']}** `{d['type']}` · {d['size']}")
    else:
        st.info("Belum ada dokumen.")

    st.divider()

    # ── Tombol Proses ──
    api_ok = bool(st.session_state.api_key and
                  st.session_state.api_key != "isi_api_key_kamu_di_sini")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Proses\nDokumen", use_container_width=True,
                     disabled=not docs_info or not api_ok):
            st.session_state.messages = []
            stats = process_documents()
            if stats:
                st.success(f" {stats['total_chunks']} chunks!")
                st.balloons()

    with col2:
        if st.button(" Muat\nTersimpan", use_container_width=True,
                     disabled=not api_ok):
            if load_existing():
                count = st.session_state.vector_store.get_document_count()
                st.success(f" {count} chunks!")
            else:
                st.error("Belum ada data")

    st.divider()

    st.session_state.show_sources = st.toggle("Tampilkan Sumber", value=True)

    if st.session_state.documents_indexed and st.session_state.vector_store:
        count = st.session_state.vector_store.get_document_count()
        st.markdown(f'<div class="status-ok"> {count} chunks terindeks</div>',
                    unsafe_allow_html=True)

    if st.button(" Hapus Riwayat Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ════════════════════════════════════════
# HALAMAN UTAMA
# ════════════════════════════════════════
st.markdown('<div class="main-header"> SAA UNKLAB </div>', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#666;'>"
    " Tanya jawab cerdas berbasis dokumen "
    "</p>",
    unsafe_allow_html=True
)

if not st.session_state.documents_indexed:
    st.info(
        "👈 **Mulai dengan langkah berikut:**\n\n"
        "1. Isi **Google Gemini API Key** di sidebar (gratis!)\n"
        "2. **Upload dokumen** (PDF/PPTX/DOCX/TXT)\n"
        "3. Klik **Proses Dokumen**\n"
        "4. Mulai bertanya! 💬"
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📕 PDF", "✅")
    col2.metric("📊 PPTX", "✅")
    col3.metric("📝 DOCX", "✅")
    col4.metric("📄 TXT", "✅")

# ── Riwayat Chat ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if (msg["role"] == "assistant" and "sources" in msg
                and st.session_state.show_sources and msg["sources"]):
            with st.expander(f"📚 Sumber ({len(msg['sources'])} referensi)"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f'<div class="source-card"><strong>Sumber {i}:</strong> '
                        f'{src["file"]} ({src["file_type"]}) | Hal. {src["page"]}<br>'
                        f'<em>{src["preview"]}</em></div>',
                        unsafe_allow_html=True
                    )

# ── Input Chat ──
if prompt := st.chat_input(
    "Ketik pertanyaan Anda...",
    disabled=not st.session_state.documents_indexed
):
    if not st.session_state.rag_chain:
        st.error("❌ RAG chain belum siap! Proses dokumen dulu.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("🔍 Mencari & memproses dengan Gemini..."):
            try:
                # Deteksi pertanyaan tentang referensi/daftar pustaka
                reference_keywords = [
                    "daftar pustaka", "referensi", "bibliografi", 
                    "sumber", "pustaka", "citation", "cited",
                    "referensi yang digunakan", "jurnal yang引用"
                ]
                is_reference_query = any(
                    keyword in prompt.lower() 
                    for keyword in reference_keywords
                )
                
                # Gunakan fallback search untuk pertanyaan tentang referensi
                if is_reference_query and hasattr(st.session_state.rag_chain, 'ask_with_fallback'):
                    result = st.session_state.rag_chain.ask_with_sources(prompt)
                    # Jika tidak ada hasil yang baik, coba fallback
                    if not result["sources"] or "tidak menemukan" in result["answer"].lower():
                        answer = st.session_state.rag_chain.ask_with_fallback(prompt)
                        sources = []
                    else:
                        answer = result["answer"]
                        sources = result["sources"]
                else:
                    result = st.session_state.rag_chain.ask_with_sources(prompt)
                    answer = result["answer"]
                    sources = result["sources"]
            except Exception as e:
                answer = f"❌ Error: {str(e)}"
                sources = []

        placeholder = st.empty()
        full = ""
        for char in answer:
            full += char
            time.sleep(0.004)
            placeholder.markdown(full + "▋")
        placeholder.markdown(full)

        if sources and st.session_state.show_sources:
            with st.expander(f"📚 Sumber ({len(sources)} referensi)"):
                for i, src in enumerate(sources, 1):
                    st.markdown(
                        f'<div class="source-card"><strong>Sumber {i}:</strong> '
                        f'{src["file"]} ({src["file_type"]}) | Hal. {src["page"]}<br>'
                        f'<em>{src["preview"]}</em></div>',
                        unsafe_allow_html=True
                    )

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "sources": sources
    })

st.divider()
st.markdown(
    "<p style='text-align:center;color:#aaa;font-size:.8rem;'>"
    ""
    "</p>",
    unsafe_allow_html=True
)
