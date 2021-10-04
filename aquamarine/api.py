import random

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from aquamarine.client import AquamarineClient

app = FastAPI()
aquamarine = AquamarineClient(embeddings="highlights1")

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


@app.get("/")
def read_root() -> dict[str, str]:
    return {"Hello": "World"}


@app.get("/highlight")
def random_highlight() -> dict[str, str]:
    highlight = random.choice(aquamarine.embeddings).text
    return {"highlight": highlight}


@app.post("/highlight")
def new_highlight(req: HighlightRequest) -> dict[str, str]:
    if req.method == "random":
        highlight = random.choice(aquamarine.embeddings).text
    elif req.method == "similar":
        query_result = aquamarine.query(
            q=req.highlight,
            embedded_content=aquamarine.embeddings,
            top_k=10,
        )
        idx = random.choice(query_result)["corpus_id"]
        highlight = aquamarine.embeddings[idx].text
    elif req.method == "different":
        query_result = aquamarine.query(
            q=req.highlight,
            embedded_content=aquamarine.embeddings,
            top_k=1500,
        )
        idx = random.choice(query_result[500:])["corpus_id"]
        highlight = aquamarine.embeddings[idx].text
    else:
        raise ValueError("Invalid method")
    return {"highlight": highlight}


def run() -> None:
    uvicorn.run("aquamarine.api:app", host="0.0.0.0", port=8000, reload=True)
