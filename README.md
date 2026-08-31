# WuBuSearch — Universal Embedding Search Engine

**WuBuSearch** is a universal embedding search engine that indexes any media (video, audio, image, text) and lets you search it with natural language.

## Features

- **Multi-backend**: NVIDIA NIM (free tier), OpenRouter (many free models), Gemini, local Qwen3-VL
- **Multi-modal**: Video, audio, image, text — all in one index
- **Highlights mode**: Auto-find the weirdest/most anomalous clips
- **SQLite storage**: No external database needed
- **Cosine similarity**: Fast numpy-based search
- **Symlinked into Big Mac**: `tools/wubusearch` → `~/Documents/wubusearch/wubusearch.py`

## Quick Start

```bash
# Install
pip install -e .

# Configure
export NVIDIA_API_KEY=nvapi-your-key
wubusearch init

# Index media
wubusearch index /path/to/videos/

# Search
wubusearch search "character laughing"

# Find weird clips
wubusearch highlights

# Stats
wubusearch stats
```

## Backends

| Backend | Free? | Models | Dims |
|---------|-------|--------|------|
| NVIDIA | Yes (tier) | llama-3.2-nv-embedqa-1b-v2 | 1024 |
| OpenRouter | Some free | text-embedding-3-small | 1536 |
| Gemini | Yes (tier) | gemini-embedding-2 | 768 |
| Local | Yes (GPU) | Qwen3-VL | varies |

## License

WaefreBeorn Umbrella v3.0 — Populace free, corps pay.
