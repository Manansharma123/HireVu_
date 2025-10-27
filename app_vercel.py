import os
import json
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import requests
import httpx
from datetime import datetime
from typing import List, Optional
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Environment variables
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "deepseek/deepseek-r1-0528-qwen3-8b:free"

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    logger.info("Serving index page.")
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/ats", response_class=HTMLResponse)
async def ats_tracker(request: Request):
    """Serve the ATS tracker page"""
    return templates.TemplateResponse("ats_tracker.html", {"request": request})

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Simple test endpoint
@app.get("/api/test")
async def test_endpoint():
    return {"message": "API is working correctly"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)