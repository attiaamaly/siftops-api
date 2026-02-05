import os
import json
import uuid
from typing import List, Dict, Any, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pypdf import PdfReader

from fastembed import TextEmbedding

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct


app = FastAPI()

# --- Config (paths in your GitHub repo) ---
MANIFEST_PATH = "siftops_dataset/docs_manifest.json"
PDF_DIR = "siftops_dataset/data/pdfs"

# --- Embedding model (Open-source, lightweight, Railway-friendly) ---
# Good default for semantic search
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
embedding_model = TextEmbedding(model_name=EMBED_MODEL_NAME)


def qdrant() -> QdrantClient:
    return QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ.get("QDRANT_API_KEY"),
        timeout=60,
    )


def chunk_text(text: str, chunk_chars: int = 900, overlap: int = 150) -> List[str]:
    """Chunk text by characters (simple + robust for PDFs)."""
    text = " ".join((text or "").split())
    if not text:
        return []
    out = []
    i = 0
    while i < len(text):
        j = min(len(text), i + chunk_chars)
        out.append(text[i:j])
        if j == len(text):
            break
        i = max(0, j - overlap)
    return out


def extract_pages(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    return [(p.extract_text() or "") for p in reader.pages]


def read_manifest() -> List[Dict[str, Any]]:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_collection(client: QdrantClient, name: str, dim: int):
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


@app.get("/")
def root():
    return {"message": "SiftOps API is running", "embedding_model": EMBED_MODEL_NAME}


@app.get("/health")
def health():
    try:
        client = qdrant()
        cols = client.get_collections()
        return {
            "status": "ok",
            "qdrant_url": os.environ.get("QDRANT_URL"),
            "using_api_key": bool(os.environ.get("QDRANT_API_KEY")),
            "collections": [c.name for c in cols.collections],
            "embedding_model": EMBED_MODEL_NAME,
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error_type": type(e).__name__, "error": str(e)},
        )


@app.post("/reindex")
def reindex(recreate: bool = True):
    """
    Reads PDFs from siftops_dataset/, chunks them, embeds locally (FastEmbed),
    and upserts into Qdrant.
    """
    try:
        client = qdrant()
        collection = os.environ.get("QDRANT_COLLECTION", "siftops_chunks")

        # Determine vector dimension once from the model output
        probe_vec = list(embedding_model.embed(["probe"]))[0]
        dim = len(probe_vec)

        if recreate:
            try:
                client.delete_collection(collection_name=collection)
            except Exception:
                pass

        ensure_collection(client, collection, dim)

        manifest = read_manifest()

        payloads: List[Dict[str, Any]] = []
        for doc in manifest:
            filename = doc.get("filename")
            if not filename:
                continue

            pdf_path = os.path.join(PDF_DIR, filename)
            if not os.path.exists(pdf_path):
                continue

            pages = extract_pages(pdf_path)
            for page_num, page_text in enumerate(pages, start=1):
                for idx, ch in enumerate(chunk_text(page_text)):
                    payloads.append(
                        {
                            "filename": filename,
                            "title": doc.get("title"),
                            "department": doc.get("department"),
                            "page": page_num,
                            "chunk_index": idx,
                            "text": ch,
                        }
                    )

        # Embed + upsert in batches
        B = 64
        upserted = 0

        for i in range(0, len(payloads), B):
            batch = payloads[i : i + B]
            texts = [p["text"] for p in batch]

            vectors = list(embedding_model.embed(texts))
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=v,
                    payload=p,
                )
                for v, p in zip(vectors, batch)
            ]

            client.upsert(collection_name=collection, points=points)
            upserted += len(points)

        return {"status": "ok", "collection": collection, "chunks_indexed": upserted}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error_type": type(e).__name__, "error": str(e)},
        )


@app.get("/search")
def search(q: str, limit: int = 5):
    """
    Semantic search: embed query locally, search Qdrant, return best chunks.
    """
    try:
        client = qdrant()
        collection = os.environ.get("QDRANT_COLLECTION", "siftops_chunks")

        qvec = list(embedding_model.embed([q]))[0]

        results = client.search(
            collection_name=collection,
            query_vector=qvec,
            limit=limit,
        )

        out = []
        for r in results:
            p = r.payload or {}
            text = p.get("text", "") or ""
            out.append(
                {
                    "score": r.score,
                    "filename": p.get("filename"),
                    "page": p.get("page"),
                    "title": p.get("title"),
                    "department": p.get("department"),
                    "snippet": (text[:300] + "…") if len(text) > 300 else text,
                }
            )

        return {"query": q, "results": out}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error_type": type(e).__name__, "error": str(e)},
        )
