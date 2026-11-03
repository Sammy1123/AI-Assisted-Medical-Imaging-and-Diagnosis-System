"""
AI-Assisted Medical Imaging and Diagnosis System
FastAPI application entry point.

⚠️  FOR EDUCATION / RESEARCH ONLY — NOT FOR CLINICAL USE ⚠️
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import CORS_ALLOW_ORIGINS

app = FastAPI(
    title="Medical Imaging AI",
    description=(
        "Pneumonia detection from chest X-rays using DenseNet-121 + Grad-CAM.\n\n"
        "**FOR EDUCATION / RESEARCH ONLY — NOT FOR CLINICAL USE.**"
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {
        "message": "Medical Imaging AI API",
        "disclaimer": "FOR EDUCATION / RESEARCH ONLY — NOT FOR CLINICAL USE.",
    }
