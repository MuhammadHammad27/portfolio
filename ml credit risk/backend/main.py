"""Backward compatibility entrypoint for backend.
Directs to the production application located in backend.app.main.
"""
from backend.app.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)