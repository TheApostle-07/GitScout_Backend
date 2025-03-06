# app/services/vector_index.py
import faiss
import numpy as np
from pymongo import MongoClient
from app.config import settings
from app.services.embedding import generate_candidate_embeddings

def build_faiss_index():
    # Connect to MongoDB using the Atlas connection details from settings
    client = MongoClient(settings.MONGO_URI)
    db = client[settings.DATABASE_NAME]
    candidates = list(db["candidates"].find())
    
    # Load the embedding model (using the model name from settings)
    model = get_embedding_model(settings.EMBEDDING_MODEL)
    
    # Generate embeddings for all candidates. This function should extract text (e.g., resume_text) from each candidate.
    embeddings = generate_candidate_embeddings(candidates, model)
    
    # Convert embeddings to a float32 NumPy array and normalize them (to use cosine similarity)
    embeddings = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    
    # Create a Faiss index using inner product on normalized vectors (which works like cosine similarity)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    return index, candidates, model