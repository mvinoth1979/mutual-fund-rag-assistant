import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import sys

# Add current directory to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the backend app
from phase_6_response_delivery.api import app

# Check if static directory exists (built by Docker)
STATIC_DIR = PROJECT_ROOT / "static"

if STATIC_DIR.exists():
    # Mount static files for the frontend
    # Everything under /static will be served as files
    # The frontend build is usually at /app/static in Docker
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")

    # Catch-all route to serve index.html for SPA routing (React)
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # If the path is /api or /chat, it should have been handled by the API routes already
        # But as a fallback, if a file exists, serve it, otherwise serve index.html
        file_path = STATIC_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(STATIC_DIR / "index.html")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
