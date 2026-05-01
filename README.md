# ✨ RAG Expert System — Google Gemini Edition

Sistem RAG untuk tanya jawab berbasis dokumen menggunakan **Google Gemini API**.

**Stack:**
- 🤖 **Google Gemini** — LLM via API (gratis, tidak perlu install apapun!)
- 🔗 **LangChain** — RAG pipeline
- 🗄️ **ChromaDB** — Vector database lokal
- 🤗 **HuggingFace** — Embedding model (all-MiniLM-L6-v2, berjalan lokal)
- 🖥️ **Streamlit** — Web UI

---

## 🚀 Setup (3 Langkah)

### 1. Dapatkan Gemini API Key (GRATIS)
1. Buka https://aistudio.google.com/app/apikey
2. Login akun Google → Klik **"Create API Key"**
3. Salin API key

### 2. Install & Konfigurasi
```bash
# Gunakan Python 3.11 di Windows
# Python 3.12 sering gagal memasang chromadb/chroma-hnswlib di Windows

# Buat virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Catatan:
# Proyek ini memakai chromadb 0.5.23; jangan ganti ke 0.3.x karena akan bentrok dengan langchain-chroma
# Di Windows, pakai Python 3.11 agar wheel chroma-hnswlib tersedia tanpa perlu Microsoft C++ Build Tools

# Isi API key di file .env
# Ganti: GOOGLE_API_KEY=isi_api_key_kamu_di_sini
# Menjadi: GOOGLE_API_KEY=AIzaSy...
```

### 3. Jalankan
```bash
streamlit run app.py
# Buka: http://localhost:8501
```

---

## 📁 Struktur Folder

```
rag-expert-system-gemini/
├── app.py                  ← Web UI Streamlit
├── cli.py                  ← Command Line
├── requirements.txt
├── .env                    ← API Key & konfigurasi
├── rag/
│   ├── document_loader.py  ← Loader PDF/PPTX/DOCX/TXT
│   ├── text_splitter.py    ← Chunking dokumen
│   ├── vector_store.py     ← ChromaDB + HuggingFace Embeddings
│   └── rag_chain.py        ← RAG pipeline (Gemini)
├── utils/helpers.py
└── data/documents/         ← Taruh dokumen di sini
```

---

## ⚙️ Konfigurasi (.env)

| Variable | Default | Keterangan |
|----------|---------|------------|
| `GOOGLE_API_KEY` | *(wajib diisi)* | API key dari Google AI Studio |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model Gemini |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Model embedding lokal |
| `CHUNK_SIZE` | `1000` | Ukuran chunk (karakter) |
| `TOP_K_RESULTS` | `5` | Jumlah dokumen yang diambil |

### Model Gemini yang direkomendasikan:
| Model | Kecepatan | Limit Gratis |
|-------|-----------|-------------|
| `gemini-2.5-flash` | ⚡ Cepat | Cek kuota terbaru di Google AI Studio |

---

## 💡 Cara Pakai

1. Upload dokumen di sidebar
2. Klik **"Proses Dokumen"**
3. Tanya di kotak chat → Gemini menjawab + tampilkan sumber

---
