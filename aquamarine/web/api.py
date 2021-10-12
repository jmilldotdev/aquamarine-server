import logging
import random
from logging.config import dictConfig

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from aquamarine.adapters.local import LocalAdapter
from aquamarine.client import AquamarineClient
from aquamarine.web import helpers
from aquamarine.web.logging import LogConfig

dictConfig(LogConfig().dict())
logger = logging.getLogger("aquamarine-web")

app = FastAPI(debug=True)
aquamarine = AquamarineClient(
    adapters=helpers.get_initial_adapters(),
    embeddings="text",
)

origins = ["http://localhost:3000", "localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HighlightRequest(BaseModel):
    highlight: str
    method: str


class QueryImagesRequest(BaseModel):
    query: str
    adapters: list[str]


class RegisterLocalAdapterRequest(BaseModel):
    path: str


@app.get("/")
def read_root() -> dict[str, str]:
    return {"Hello": "World"}


@app.get("/adapters")
def get_adapters() -> dict[str, list]:
    adapter_aliases = list(aquamarine.adapters.keys())
    return {"adapters": adapter_aliases}


@app.post("/register_adapter")
def register_adapter(req: RegisterLocalAdapterRequest) -> dict[str, str]:
    adapter = LocalAdapter(path=req.path)
    aquamarine.adapters[adapter.alias] = adapter
    return {"status": "ok"}


@app.get("/highlight")
def random_highlight() -> dict[str, str]:
    highlight = random.choice(aquamarine.embeddings)
    resp = highlight.__dict__
    del resp["embedding"]
    return {"highlight": resp}


@app.post("/highlight")
def new_highlight(req: HighlightRequest) -> dict[str, dict]:
    if req.method == "random":
        highlight = random.choice(aquamarine.embeddings)
    elif req.method == "similar":
        query_result = aquamarine.query(q=req.highlight, top_k=10)
        highlight = random.choice(query_result).content
    elif req.method == "different":
        query_result = aquamarine.query(q=req.highlight, top_k=1500)
        highlight = random.choice(query_result[1000:]).content
    else:
        raise ValueError("Invalid method")
    resp = highlight.__dict__
    return {"highlight": resp}


@app.post("/query_images")
def query_images(req: QueryImagesRequest) -> dict[str, list]:
    if not aquamarine.image_model:
        aquamarine.load_models()
    selected_adapters = helpers.get_selected_adapters(aquamarine, req.adapters)
    query_result = aquamarine.query(
        q=req.query,
        model=aquamarine.image_model,
        adapters=selected_adapters,
    )
    return {"images": [image.content.path for image in query_result]}


@app.get("/scatter_data")
def get_scatter_data() -> dict[str, list]:
    res = aquamarine.load_tsne()
    tsne_data = {
        "scatterData": [],
        "highlights": [
            v.__dict__
            for k, v in aquamarine.adapters["Atomic Notes"].embedded_content.items()
        ],
    }
    for r in res:
        tsne = r[1].tolist()
        # highlight = r[0].__dict__
        tsne_data["scatterData"].append(
            {"x": tsne[0], "y": tsne[1], "id": r[0].corpus_id},
        )
    return tsne_data


def run() -> None:
    uvicorn.run(
        "aquamarine.web.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        debug=True,
    )
