import os, json, uuid
from typing import List, Dict, Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pypdf import PdfReader
from fastembed import TextEmbedding

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

app = FastAPI()

# --------- setup ---------
embedding_model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

def qdrant() -> QdrantClient:
    return QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ.get("QDRANT_API_KEY"),
        timeout=60,
    )

# --------- helpers ---------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    text = " ".join((text or "").split())
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i + chunk_size]
        if chunk:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def extract_pages(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    return [(p.extract_text() or "") for p in reader.pages]

def read_manifest() -> List[Dict[str, Any]]:
    with open("siftops_dataset/docs_manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_collection(client: QdrantClient, name: str, dim: int):
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

# --------- routes ---------
@app.get("/")
def root():
    return {"message": "SiftOps API is running"}

@app.get("/health")
def health():
    try:
        cols = qdrant().get_collections()
        return {
            "status": "ok",
            "collections": [c.name for c in cols.collections],
            "embedding_model": MODEL_NAME,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/reindex")
def reindex(recreate: bool = True):
    client = qdrant()
    collection = os.environ.get("QDRANT_COLLECTION", "siftops_chunks")

    dim = embedding_model.get_sentence_embedding_dimension()

    if recreate:
        try:
            client.delete_collection(collection)
        except Exception:
            pass

    ensure_collection(client, collection, dim)

    manifest = read_manifest()
    pdf_dir = "siftops_dataset/data/pdfs"

    points = []
    for doc in manifest:
        pdf_path = os.path.join(pdf_dir, doc["filename"])
        if not os.path.exists(pdf_path):
            continue

        pages = extract_pages(pdf_path)
        for page_num, page_text in enumerate(pages, start=1):
            chunks = chunk_text(page_text)
            embeddings = list(embedding_model.embed(chunks))

            for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "filename": doc["filename"],
                            "title": doc.get("title"),
                            "department": doc.get("department"),
                            "page": page_num,
                            "chunk_index": idx,
                            "text": chunk,
                        },
                    )
                )

    client.upsert(collection_name=collection, points=points)

    return {
        "status": "ok",
        "collection": collection,
        "chunks_indexed": len(points),
    }

@app.get("/search")
def search(q: str, limit: int = 5):
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
        text = p.get("text", "")
        out.append({
            "score": r.score,
            "filename": p.get("filename"),
            "page": p.get("page"),
            "title": p.get("title"),
            "snippet": (text[:300] + "…") if len(text) > 300 else text,
        })

    return {"query": q, "results": out}
