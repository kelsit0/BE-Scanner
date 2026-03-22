from typing import Union

from fastapi import FastAPI
from app.api import analyze, auth
from app.core.config import init_db

from fastapi.middleware.cors import CORSMiddleware

from app.models.analysis import Analysis

app = FastAPI()

app.include_router(analyze.router, prefix="/api")

app.include_router(auth.router, prefix="/auth")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/")
async def root():
    return {"message": "Radiografía Analyzer API lista 🚀"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}