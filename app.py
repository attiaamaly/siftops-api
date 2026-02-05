import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient

app = FastAPI()

@app.get("/")
def root():
    return {"message": "SiftOps API is running"}

@app.get("/health")
def health():
    url = os.environ.get("QDRANT_URL")
    api_key = os.environ.get("QDRANT_API_KEY")

    try:
        if not url:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "reason": "QDRANT_URL is missing"},
            )

        client = QdrantClient(url=url, api_key=(api_key or None))
        collections = client.get_collections()

        return {
            "status": "ok",
            "qdrant_url": url,
            "using_api_key": bool(api_key),
            "collections": [c.name for c in collections.collections],
        }

    except Exception as e:
        # This will show the REAL reason (401/403/timeout/etc.)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "qdrant_url": url,
                "using_api_key": bool(api_key),
                "error_type": type(e).__name__,
                "error": str(e),
            },
        )
