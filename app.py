import os
from fastapi import FastAPI
from qdrant_client import QdrantClient

app = FastAPI()

def get_qdrant() -> QdrantClient:
    url = os.environ["QDRANT_URL"]
    api_key = os.environ.get("QDRANT_API_KEY") or None
    return QdrantClient(url=url, api_key=api_key)

@app.get("/health")
def health():
    # If Qdrant is reachable, this works
    qdrant = get_qdrant()
    collections = qdrant.get_collections()
    return {
        "status": "ok",
        "qdrant": "reachable",
        "collections": [c.name for c in collections.collections],
    }
