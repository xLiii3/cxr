"""HTTP API around the same engine.

    uvicorn api:app --port 8000
    curl -F "file=@examples/cardiomegaly.png" http://127.0.0.1:8000/predict
"""
from io import BytesIO

from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image

from src.service import Engine

app = FastAPI(title="CXR Multimodal API", version="1.0",
              description="14-class chest X-ray classification, text and image retrieval")
ENGINE = Engine()


def load(data):
    return Image.open(BytesIO(data)).convert("RGB")


@app.get("/health")
def health():
    return {"status": "ok", "gallery": len(ENGINE.gallery)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    return {"predictions": ENGINE.predict(load(await file.read()))}


@app.post("/search/text")
async def search_text(text: str = Form(...), k: int = Form(5)):
    return {"results": [{k2: v for k2, v in r.items() if k2 != "image"}
                        for r in ENGINE.search_text(text, k)]}


@app.post("/search/image")
async def search_image(file: UploadFile = File(...), k: int = Form(5)):
    return {"results": [{k2: v for k2, v in r.items() if k2 != "image"}
                        for r in ENGINE.search_image(load(await file.read()), k)]}
