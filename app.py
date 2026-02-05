import os, json, uuid
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pypdf import PdfReader
from openai import OpenAI

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

app = FastAPI()

def env(name: str, default: Optional[str] = None) -> str:
    v = os.environ.get(name, default)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v

def qdrant() -> QdrantClient:
    return QdrantClient(
        url=env("QDRANT_URL"),
        api_key=os.environ.get("QDRANT_API_KEY") or None,
        timeout=60,
    )

def oai() -> OpenAI:
    return OpenAI(api_key=env("OPENAI_API_KEY"))

def chunk_text(text: str, chunk_chars: int = 1200, overlap: int = 200) -> List[str]:
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

def ensure_collection(client: QdrantClient, name: str, dim: int):
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

def read_manifest() -> List[Dict[str, Any]]:
    with open("siftops_dataset/docs_manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)

def extract_pages(pdf_path: str) -> List[str]:
    r = PdfReader(pdf_path)
    return [(p.extract_text() or "") for p in r.pages]

@app.get("/")
def root():
    return {"message": "SiftOps API is running"}

@app.get("/health")
def health():
    try:
        c = qdrant()
        cols = c.get_collections()
        return {
            "status": "ok",
            "qdrant_url": env("QDRANT_URL"),
            "using_api_key": bool(os.environ.get("QDRANT_API_KEY")),
            "collections": [x.name for x in cols.collections],
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error_type": type(e).__name__, "error": str(e)},
        )

@app.post("/reindex")
def reindex(recreate: bool = True):
    client = qdrant()
    openai = oai()
    collection = os.environ.get("QDRANT_COLLECTION", "siftops_chunks")
    model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

    # infer embedding dimension
    dim = len(openai.embeddings.create(model=model, input=["probe"]).data[0].embedding)

    if recreate:
        try:
            client.delete_collection(collection_name=collection)
        except Exception:
            pass

    ensure_collection(client, collection, dim)

    manifest = read_manifest()
    pdf_dir = "siftops_dataset/data/pdfs"

    payloads = []
    for doc in manifest:
        pdf_path = os.path.join(pdf_dir, doc["filename"])
        if not os.path.exists(pdf_path):
            continue
        pages = extract_pages(pdf_path)
        for page_num, page_text in enumerate(pages, start=1):
            for idx, ch in enumerate(chunk_text(page_text)):
                payloads.append({
                    "filename": doc["filename"],
                    "title": doc.get("title"),
                    "department": doc.get("department"),
                    "page": page_num,
                    "chunk_index": idx,
                    "text": ch,
                })

    # batch embed + upsert
    B = 64
    upserted = 0
    for i in range(0, len(payloads), B):
        batch = payloads[i:i+B]
        texts = [p["text"] for p in batch]
        emb = openai.embeddings.create(model=model, input=texts)
        vectors = [e.embedding for e in emb.data]
        points = [PointStruct(id=str(uuid.uuid4()), vector=v, payload=p)
                  for v, p in zip(vectors, batch)]
        client.upsert(collection_name=collection, points=points)
        upserted += len(points)

    return {"status": "ok", "collection": collection, "chunks_indexed": upserted}

@app.get("/search")
def search(q: str, limit: int = 5):
    client = qdrant()
    openai = oai()
    collection = os.environ.get("QDRANT_COLLECTION", "siftops_chunks")
    model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

    qvec = openai.embeddings.create(model=model, input=[q]).data[0].embedding

    res = client.search(
        collection_name=collection,
        query_vector=qvec,
        limit=limit,
    )

    out = []
    for r in res:
        p = r.payload or {}
        text = p.get("text", "")
        out.append({
            "score": r.score,
            "filename": p.get("filename"),
            "page": p.get("page"),
            "title": p.get("title"),
            "snippet": (text[:300] + "…") if len(text) > 300 else text,
        })

    return {"query": q, "results": out}
