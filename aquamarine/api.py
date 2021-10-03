import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aquamarine.client import AquamarineClient

app = FastAPI()
aquamarine = AquamarineClient()

origins = ["http://localhost:3000", "localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"Hello": "World"}


def run():
    uvicorn.run("aquamarine.api:app", host="0.0.0.0", port=8000, reload=True)
