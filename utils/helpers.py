"""
utils/helpers.py - Gemini Version
"""

import os
import requests
from pathlib import Path
from typing import List, Optional

GEMINI_MODELS = {
    "gemini-2.5-flash": "Default yang aman untuk RAG",
    "gemini-2.5-flash-lite": "Lebih ringan dan hemat",
    "gemini-2.5-pro": "Lebih kuat untuk pertanyaan kompleks",
}

DEPRECATED_GEMINI_MODELS = {
    "gemini-1.5-flash": "gemini-2.5-flash",
    "gemini-1.5-pro": "gemini-2.5-pro",
    "gemini-1.5-flash-8b": "gemini-2.5-flash-lite",
    "gemini-2.0-flash": "gemini-2.5-flash",
    "gemini-2.0-flash-lite": "gemini-2.5-flash-lite",
}


def check_gemini_api_key(api_key: str = None) -> bool:
    key = api_key or os.getenv("GOOGLE_API_KEY", "")
    if not key or key == "isi_api_key_kamu_di_sini":
        return False
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
        response = requests.get(url, timeout=10)
        return response.status_code == 200
    except Exception:
        return False


def get_gemini_models() -> dict:
    return GEMINI_MODELS


def normalize_gemini_model(model_name: Optional[str]) -> str:
    if not model_name:
        return "gemini-2.5-flash"
    return DEPRECATED_GEMINI_MODELS.get(model_name, model_name)


def format_file_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def get_documents_info(directory: str) -> List[dict]:
    supported = {".pdf", ".pptx", ".ppt", ".docx", ".txt"}
    directory_path = Path(directory)
    if not directory_path.exists():
        return []
    files_info = []
    for file_path in directory_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported:
            stat = file_path.stat()
            files_info.append({
                "name": file_path.name,
                "path": str(file_path),
                "type": file_path.suffix.upper().replace(".", ""),
                "size": format_file_size(stat.st_size),
                "size_bytes": stat.st_size,
            })
    return sorted(files_info, key=lambda x: x["name"])


def save_uploaded_file(uploaded_file, save_dir: str) -> str:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    file_path = save_path / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path)


GEMINI_API_KEY_GUIDE = """
Cara mendapatkan Gemini API Key (GRATIS):
1. Buka: https://aistudio.google.com/app/apikey
2. Login dengan akun Google
3. Klik "Create API Key"
4. Paste ke file .env: GOOGLE_API_KEY=your_key_here

Gunakan model yang masih aktif, misalnya: gemini-2.5-flash
"""
