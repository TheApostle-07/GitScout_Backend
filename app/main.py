import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import candidate, github, job  # Import the new job routes
from app.utils.logger import setup_logger

# Initialize FastAPI app with metadata
app = FastAPI(
    title="GitScout API",
    description="API for GitScout - an AI-driven recruiting tool using GitHub commit embeddings.",
    version="1.0.0"
)

# Set up CORS (allowing all origins here; modify as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify a list of allowed origins, e.g. ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up custom logging
setup_logger()

# Include routers from our routes modules
app.include_router(candidate.router, prefix="/candidate", tags=["Candidate"])
app.include_router(github.router, prefix="/github", tags=["GitHub"])
app.include_router(job.router, prefix="/job", tags=["Job"])

# Root endpoint for a simple health check
@app.get("/")
def read_root():
    return {"message": "Welcome to GitScout Backend"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)