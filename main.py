import os
from app import app

if __name__ == "__main__":
    # Vercel will automatically handle the server setup
    # This is just for local development
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)