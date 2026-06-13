"""
MappingMind — FastAPI Production API
======================================
Exposes RAG pipeline as REST endpoints.
Run: uvicorn api.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="MappingMind API",
    description="Enterprise Data Mapping Intelligence — Powered by Agentic RAG",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "MappingMind"}


@app.get("/")
def root():
    return {
        "service": "MappingMind",
        "version": "1.0.0",
        "problem": "Data mapping institutional knowledge — instantly queryable",
        "docs": "/docs"
    }