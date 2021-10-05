import random

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from aquamarine.client import AquamarineClient

app = FastAPI()
aquamarine = AquamarineClient(embeddings="highlights2")


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


def run() -> None:
    uvicorn.run("aquamarine.api:app", host="0.0.0.0", port=8000, reload=True)
