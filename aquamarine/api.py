from fastapi import FastAPI

from aquamarine.client import AquamarineClient

app = FastAPI()
aquamarine = AquamarineClient()


@app.get("/")
def read_root() -> dict[str, str]:
    return {"Hello": "World"}
