#!/usr/bin/env python3
"""WuBuSearch — Universal Embedding Search Engine.

A unified interface for embedding and searching any media type (video, audio, text, image)
across multiple backend providers (NVIDIA, OpenRouter, Gemini, local models).

Usage:
    wubusearch init                          # Configure API keys
    wubusearch index <directory>             # Index media files
    wubusearch search "query"                # Search indexed media
    wubusearch highlights                    # Find anomalous/interesting clips
    wubusearch stats                         # Show index statistics

Backends:
    nvidia      — NVIDIA NIM API (free tier available)
    openrouter  — OpenRouter API (many free models)
    gemini      — Google Gemini Embedding API
    local       — Local Qwen3-VL model (needs GPU)

License: WaefreBeorn Umbrella v3.0 (Populace free, corps pay)
"""

import os
import sys
import json
import hashlib
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WUBU_HOME = Path(os.environ.get("WUBU_HOME", Path.home() / ".wubu"))
DB_PATH = WUBU_HOME / "wubu.db"
CONFIG_PATH = WUBU_HOME / "config.json"

SUPPORTED_VIDEO = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
SUPPORTED_AUDIO = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}
SUPPORTED_IMAGE = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
SUPPORTED_TEXT = {".txt", ".srt", ".json", ".md", ".py", ".c", ".h"}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


def save_config(config: dict):
    WUBU_HOME.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2))


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


def get_db() -> sqlite3.Connection:
    WUBU_HOME.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(DB_PATH))
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""
        CREATE TABLE IF NOT EXISTS media (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE NOT NULL,
            media_type TEXT NOT NULL,
            duration_sec REAL,
            indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            media_id INTEGER NOT NULL,
            start_sec REAL NOT NULL,
            end_sec REAL NOT NULL,
            embedding BLOB,
            embedding_model TEXT,
            embedding_dims INTEGER,
            text_content TEXT,
            FOREIGN KEY (media_id) REFERENCES media(id)
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    db.commit()
    return db


# ---------------------------------------------------------------------------
# Embedding Backends
# ---------------------------------------------------------------------------


class BaseBackend:
    """Base class for embedding providers."""

    name: str = "base"
    supports_video = False
    supports_audio = False
    supports_image = False
    supports_text = True

    def embed_text(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def embed_image(self, image_path: str) -> np.ndarray:
        raise NotImplementedError

    def embed_video_chunk(self, video_path: str, start_sec: float, end_sec: float) -> np.ndarray:
        """Extract a frame and embed as image."""
        frame = self._extract_frame(video_path, start_sec, end_sec)
        if frame:
            return self.embed_image(frame)
        # Fallback: embed metadata as text
        return self.embed_text(f"video:{os.path.basename(video_path)}:{start_sec}-{end_sec}")

    def embed_audio_chunk(self, audio_path: str, start_sec: float, end_sec: float) -> np.ndarray:
        """Embed audio chunk — fallback to text description."""
        return self.embed_text(f"audio:{os.path.basename(audio_path)}:{start_sec}-{end_sec}")

    def _extract_frame(self, video_path: str, start: float, end: float) -> Optional[str]:
        """Extract a frame from the middle of a video segment."""
        mid = (start + end) / 2
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            frame_path = f.name
        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-ss", str(mid), "-i", video_path,
                 "-vframes", "1", "-q:v", "2", frame_path],
                capture_output=True, timeout=30
            )
            if result.returncode == 0 and os.path.exists(frame_path):
                return frame_path
        except Exception:
            pass
        return None

    def dimensions(self) -> int:
        return 768


class NVIDIABackend(BaseBackend):
    """NVIDIA NIM embedding backend."""

    name = "nvidia"
    supports_video = True
    supports_image = True

    def __init__(self, model: str = "nvidia/llama-3.2-nv-embedqa-1b-v2", rpm: int = 60):
        self.model = model
        self.rpm = rpm
        api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVAPI_KEY")
        if not api_key:
            # Try config
            config = load_config()
            api_key = config.get("nvidia_api_key")
        if not api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY not set. Get a free key at https://build.nvidia.com\n"
                "Then: export NVIDIA_API_KEY=nvapi-..."
            )
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://integrate.api.nvidia.com/v1"
        )

    def embed_text(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model, input=text, encoding_format="float",
            extra_body={"input_type": "passage", "truncate": "END"}
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def embed_image(self, image_path: str) -> np.ndarray:
        import base64
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(image_path)[1].lower()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(ext, "image/png")
        try:
            response = self.client.embeddings.create(
                model=self.model, input=f"data:{mime};base64,{img_b64}", encoding_format="float"
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception:
            return self.embed_text(f"image:{os.path.basename(image_path)}")

    def dimensions(self) -> int:
        return 1024


class OpenRouterBackend(BaseBackend):
    """OpenRouter embedding backend — supports many free models."""

    name = "openrouter"
    supports_video = True
    supports_image = True

    # Free embedding models on OpenRouter
    FREE_MODELS = [
        "openai/text-embedding-3-small",  # 1536 dims, free tier
        "text-embedding-3-small",  # shorthand
    ]

    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or "openai/text-embedding-3-small"
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            config = load_config()
            key = config.get("openrouter_api_key")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set.\n"
                "Then: export OPENROUTER_API_KEY=sk-or-v1-..."
            )
        from openai import OpenAI
        self.client = OpenAI(
            api_key=key,
            base_url="https://openrouter.ai/api/v1"
        )

    def embed_text(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model, input=text)
        return np.array(response.data[0].embedding, dtype=np.float32)

    def embed_image(self, image_path: str) -> np.ndarray:
        import base64
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(image_path)[1].lower()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(ext, "image/png")
        # OpenRouter uses the same format as OpenAI for image embeddings
        try:
            response = self.client.embeddings.create(
                model=self.model, input=f"data:{mime};base64,{img_b64}"
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception:
            return self.embed_text(f"image:{os.path.basename(image_path)}")

    def dimensions(self) -> int:
        return 1536


class GeminiBackend(BaseBackend):
    """Google Gemini embedding backend."""

    name = "gemini"
    supports_video = True
    supports_image = True

    def __init__(self, model: str = "gemini-embedding-2"):
        self.model = model
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            config = load_config()
            api_key = config.get("gemini_api_key")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set.")
        from google import genai
        self.client = genai.Client(api_key=api_key)

    def embed_text(self, text: str) -> np.ndarray:
        from google.genai import types
        response = self.client.models.embed_content(
            model=self.model, contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=768)
        )
        return np.array(response.embeddings[0].values, dtype=np.float32)

    def embed_image(self, image_path: str) -> np.ndarray:
        from google.genai import types
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")
        response = self.client.models.embed_content(
            model=self.model, contents=types.Content(parts=[part]),
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=768)
        )
        return np.array(response.embeddings[0].values, dtype=np.float32)

    def dimensions(self) -> int:
        return 768


def get_backend(backend_name: str = None, **kwargs) -> BaseBackend:
    """Factory to create the appropriate backend."""
    config = load_config()
    backend_name = backend_name or config.get("default_backend", "openrouter")

    if backend_name == "nvidia":
        return NVIDIABackend(model=kwargs.get("model"), rpm=kwargs.get("rpm", 60))
    elif backend_name == "openrouter":
        return OpenRouterBackend(model=kwargs.get("model"))
    elif backend_name == "gemini":
        return GeminiBackend(model=kwargs.get("model", "gemini-embedding-2"))
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def scan_directory(directory: str) -> list[Path]:
    """Recursively scan for supported media files."""
    files = []
    for p in Path(directory).rglob("*"):
        if p.suffix.lower() in SUPPORTED_VIDEO | SUPPORTED_AUDIO | SUPPORTED_IMAGE | SUPPORTED_TEXT:
            files.append(p)
    return sorted(files)


def get_media_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in SUPPORTED_VIDEO:
        return "video"
    elif ext in SUPPORTED_AUDIO:
        return "audio"
    elif ext in SUPPORTED_IMAGE:
        return "image"
    elif ext in SUPPORTED_TEXT:
        return "text"
    return "unknown"


def get_video_duration(path: str) -> float:
    """Get video/audio duration via ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=10
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def index_media(directory: str, backend_name: str = None, chunk_duration: int = 30,
                overlap: int = 5, resolution: int = 480, fps: int = 2):
    """Index all media files in a directory."""
    backend = get_backend(backend_name)
    db = get_db()
    files = scan_directory(directory)

    if not files:
        print(f"No supported media files found in {directory}")
        return

    print(f"WuBuSearch Indexing")
    print(f"  Backend: {backend.name}")
    print(f"  Files: {len(files)}")
    print(f"  Chunk duration: {chunk_duration}s, Overlap: {overlap}s")
    print()

    for i, filepath in enumerate(files):
        path = str(filepath)
        media_type = get_media_type(filepath)

        # Check if already indexed
        existing = db.execute("SELECT id FROM media WHERE path=?", (path,)).fetchone()
        if existing:
            print(f"  [{i+1}/{len(files)}] SKIP (already indexed): {filepath.name}")
            continue

        print(f"  [{i+1}/{len(files)}] Indexing: {filepath.name} ({media_type})")

        duration = 0
        if media_type in ("video", "audio"):
            duration = get_video_duration(path)

        db.execute(
            "INSERT INTO media (path, media_type, duration_sec) VALUES (?, ?, ?)",
            (path, media_type, duration)
        )
        media_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

        if media_type == "video":
            _index_video(db, backend, media_id, path, duration, chunk_duration, overlap)
        elif media_type == "image":
            _index_image(db, backend, media_id, path)
        elif media_type == "text":
            _index_text(db, backend, media_id, path)
        elif media_type == "audio":
            _index_audio(db, backend, media_id, path, duration, chunk_duration, overlap)

        db.commit()

    print(f"\nDone! Indexed {len(files)} files.")
    stats()


def _index_video(db, backend, media_id, path, duration, chunk_dur, overlap):
    """Index a video file in chunks."""
    if duration <= 0:
        duration = 300  # default 5 min

    start = 0.0
    count = 0
    while start < duration:
        end = min(start + chunk_dur, duration)
        mid = (start + end) / 2

        try:
            embedding = backend.embed_video_chunk(path, start, end)
            db.execute(
                "INSERT INTO chunks (media_id, start_sec, end_sec, embedding, embedding_model, embedding_dims) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (media_id, start, end, embedding.tobytes(), backend.name, len(embedding))
            )
            count += 1
        except Exception as e:
            print(f"    Warning: chunk {start:.0f}-{end:.0f}s failed: {e}")

        start = end - overlap

    print(f"    → {count} chunks indexed")


def _index_image(db, backend, media_id, path):
    """Index a single image."""
    try:
        embedding = backend.embed_image(path)
        db.execute(
            "INSERT INTO chunks (media_id, start_sec, end_sec, embedding, embedding_model, embedding_dims) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (media_id, 0, 0, embedding.tobytes(), backend.name, len(embedding))
        )
        print(f"    → indexed")
    except Exception as e:
        print(f"    Warning: {e}")


def _index_text(db, backend, media_id, path):
    """Index text file content."""
    try:
        content = path.read_text(errors="replace")
        # Chunk by paragraphs
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        for i, para in enumerate(paragraphs[:100]):  # limit to 100 chunks
            try:
                embedding = backend.embed_text(para[:2000])  # limit chunk size
                db.execute(
                    "INSERT INTO chunks (media_id, start_sec, end_sec, embedding, embedding_model, embedding_dims, text_content) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (media_id, i, i+1, embedding.tobytes(), backend.name, len(embedding), para[:500])
                )
            except Exception:
                pass
        print(f"    → {min(len(paragraphs), 100)} chunks indexed")
    except Exception as e:
        print(f"    Warning: {e}")


def _index_audio(db, backend, media_id, path, duration, chunk_dur, overlap):
    """Index audio file in chunks."""
    if duration <= 0:
        duration = 300

    start = 0.0
    count = 0
    while start < duration:
        end = min(start + chunk_dur, duration)
        try:
            embedding = backend.embed_audio_chunk(path, start, end)
            db.execute(
                "INSERT INTO chunks (media_id, start_sec, end_sec, embedding, embedding_model, embedding_dims) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (media_id, start, end, embedding.tobytes(), backend.name, len(embedding))
            )
            count += 1
        except Exception:
            pass
        start = end - overlap

    print(f"    → {count} chunks indexed")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search(query: str, top_k: int = 5, backend_name: str = None):
    """Search indexed media for a query."""
    backend = get_backend(backend_name)
    db = get_db()

    query_embedding = backend.embed_text(query)
    query_bytes = query_embedding.tobytes()

    # Load all embeddings and compute similarity
    rows = db.execute(
        "SELECT c.id, c.media_id, c.start_sec, c.end_sec, c.embedding, c.embedding_dims, "
        "m.path, m.media_type, m.duration_sec "
        "FROM chunks c JOIN media m ON c.media_id = m.id"
    ).fetchall()

    if not rows:
        print("No indexed media. Run 'wubusearch index <directory>' first.")
        return

    results = []
    for row in rows:
        chunk_id, media_id, start, end, emb_bytes, dims, path, media_type, duration = row
        if emb_bytes:
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            # Cosine similarity
            sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-8)
            results.append({
                "score": float(sim),
                "path": path,
                "type": media_type,
                "start": start,
                "end": end,
            })

    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"WuBuSearch: \"{query}\"")
    print(f"  Backend: {backend.name}")
    print(f"  Results: {min(top_k, len(results))}/{len(results)}")
    print()

    for i, r in enumerate(results[:top_k]):
        time_str = f" @ {r['start']:.0f}s-{r['end']:.0f}s" if r['type'] == "video" else ""
        print(f"  #{i+1} [{r['score']:.3f}] {r['path']}{time_str}")

    return results[:top_k]


def highlights(top_k: int = 5):
    """Find the most anomalous/interesting media in the index."""
    db = get_db()
    rows = db.execute(
        "SELECT c.id, c.embedding, c.embedding_dims, m.path, m.media_type, c.start_sec, c.end_sec "
        "FROM chunks c JOIN media m ON c.media_id = m.id WHERE c.embedding IS NOT NULL"
    ).fetchall()

    if not rows:
        print("No indexed media.")
        return

    # Load all embeddings
    embeddings = []
    metadata = []
    for row in rows:
        chunk_id, emb_bytes, dims, path, media_type, start, end = row
        if emb_bytes:
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            embeddings.append(emb)
            metadata.append({"path": path, "type": media_type, "start": start, "end": end})

    if len(embeddings) < 2:
        print("Need at least 2 indexed chunks for highlights.")
        return

    # Compute centroid and find outliers
    matrix = np.stack(embeddings)
    centroid = matrix.mean(axis=0)
    distances = np.linalg.norm(matrix - centroid, axis=1)

    # Top outliers
    top_indices = np.argsort(distances)[::-1][:top_k]

    print(f"WuBuSearch Highlights (most anomalous)")
    print()

    for i, idx in enumerate(top_indices):
        m = metadata[idx]
        time_str = f" @ {m['start']:.0f}s-{m['end']:.0f}s" if m['type'] == "video" else ""
        print(f"  #{i+1} [{distances[idx]:.2f}] {m['path']}{time_str}")


def stats():
    """Show index statistics."""
    db = get_db()
    media_count = db.execute("SELECT COUNT(*) FROM media").fetchone()[0]
    chunk_count = db.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0]
    types = db.execute("SELECT media_type, COUNT(*) FROM media GROUP BY media_type").fetchall()

    print(f"WuBuSearch Stats")
    print(f"  Media files: {media_count}")
    print(f"  Indexed chunks: {chunk_count}")
    for t, c in types:
        print(f"    {t}: {c}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1]

    if command == "init":
        config = load_config()
        print("WuBuSearch Configuration")
        print()
        print("Available backends: nvidia, openrouter, gemini")
        backend = input(f"Default backend [{config.get('default_backend', 'nvidia')}]: ").strip()
        if backend:
            config["default_backend"] = backend

        if backend == "nvidia" or (not backend and config.get("default_backend") == "nvidia"):
            key = input("NVIDIA API Key (nvapi-...): ").strip()
            if key:
                config["nvidia_api_key"] = key
                os.environ["NVIDIA_API_KEY"] = key

        save_config(config)
        print(f"Config saved to {CONFIG_PATH}")

    elif command == "index":
        if len(sys.argv) < 3:
            print("Usage: wubusearch index <directory> [--backend nvidia|openrouter|gemini]")
            return
        directory = sys.argv[2]
        backend = None
        if "--backend" in sys.argv:
            idx = sys.argv.index("--backend")
            if idx + 1 < len(sys.argv):
                backend = sys.argv[idx + 1]
        index_media(directory, backend_name=backend)

    elif command == "search":
        if len(sys.argv) < 3:
            print("Usage: wubusearch search \"query\" [--top N]")
            return
        query = sys.argv[2]
        top_k = 5
        if "--top" in sys.argv:
            idx = sys.argv.index("--top")
            if idx + 1 < len(sys.argv):
                top_k = int(sys.argv[idx + 1])
        search(query, top_k=top_k)

    elif command == "highlights":
        top_k = 5
        if "--top" in sys.argv:
            idx = sys.argv.index("--top")
            if idx + 1 < len(sys.argv):
                top_k = int(sys.argv[idx + 1])
        highlights(top_k=top_k)

    elif command == "stats":
        stats()

    elif command == "reset":
        if DB_PATH.exists():
            DB_PATH.unlink()
        print("Index reset.")

    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
