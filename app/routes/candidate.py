from fastapi import APIRouter, HTTPException, Body
from pymongo import MongoClient
from bson import ObjectId
from app.config import settings
from app.services.vector_index import build_faiss_index
from app.services.embedding import generate_ideal_candidate_vector
from faiss import normalize_L2
import numpy as np

router = APIRouter()

# MongoDB client for candidate collection
client = MongoClient(settings.MONGO_URI)
db = client[settings.DATABASE_NAME]
candidates_collection = db["candidates"]

# -------------------------------
# Candidate Matching Endpoint
# -------------------------------
@router.post("/match")
async def match_candidates(criteria: dict = Body(...)):
    """
    Accepts ideal candidate criteria as JSON (for example, {"languages": ["Python"], "min_commits": 50}).
    It generates an ideal candidate vector from these criteria, adjusts its dimension to match the candidate vectors,
    and then computes cosine similarity against all candidate vectors in the Faiss index.
    Returns the top matching candidates along with a hire decision.
    """
    # Build (or load) the Faiss index along with candidate documents and the embedding model.
    index, candidates, model = build_faiss_index()
    
    # Generate the ideal candidate vector using recruiter criteria.
    # This function returns a vector of dimension, say, 768.
    ideal_vector = generate_ideal_candidate_vector(criteria)
    
    # Candidate vectors are built using generate_candidate_vector, which concatenates a numeric feature.
    # For example, if the text vector is 768-dim and you append one numeric feature,
    # candidate vectors will be 769-dimensional. Adjust the ideal vector accordingly.
    ideal_vector = np.concatenate((ideal_vector, np.array([0.0], dtype=float)))
    ideal_vector = np.array([ideal_vector], dtype=np.float32)  # shape: (1, dimension)
    normalize_L2(ideal_vector)
    
    # Determine how many top candidates to return
    top_k = min(5, len(candidates))
    
    # Search the index using the ideal vector
    distances, indices = index.search(ideal_vector, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        candidate = candidates[idx]
        candidate['_id'] = str(candidate['_id'])
        similarity_score = float(distances[0][i])
        # Use a predefined threshold from settings for the decision (e.g., 0.8)
        hire_decision = "Hire" if similarity_score >= settings.SIMILARITY_THRESHOLD else "No-Hire"
        results.append({
            "candidate_id": candidate['_id'],
            "name": candidate.get("name", "N/A"),
            "similarity_score": similarity_score,
            "hire_decision": hire_decision
        })
    
    return {"results": results}

# -------------------------------
# Existing Candidate Endpoints
# -------------------------------
@router.get("/search")
async def search_candidates(q: str):
    # Build (or load) the Faiss index and candidate embeddings
    index, candidates, model = build_faiss_index()
    
    # Generate the embedding for the job description (with dimension matching candidate vectors)
    job_text_vector = model.encode([q])[0]
    # Append a numeric feature (default 0) to match candidate vector dimension
    job_embedding = np.concatenate((job_text_vector, np.array([0], dtype=float)))
    job_embedding = np.array([job_embedding], dtype=np.float32)  # shape: (1, dimension)
    normalize_L2(job_embedding)
    
    # Use the lesser of 5 or the actual number of candidates
    top_k = min(5, len(candidates))
    
    distances, indices = index.search(job_embedding, top_k)
    
    results = []
    for idx in indices[0]:
        candidate = candidates[idx]
        candidate['_id'] = str(candidate['_id'])
        results.append(candidate)
    
    return {"results": results, "distances": distances.tolist()}

@router.post("/")
async def create_candidate(candidate: dict):
    result = candidates_collection.insert_one(candidate)
    candidate['_id'] = str(result.inserted_id)
    return candidate

@router.get("/")
async def get_all_candidates():
    candidates = list(candidates_collection.find())
    for candidate in candidates:
        candidate['_id'] = str(candidate['_id'])
    return candidates

@router.get("/{candidate_id}")
async def get_candidate(candidate_id: str):
    candidate = candidates_collection.find_one({"_id": ObjectId(candidate_id)})
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    candidate['_id'] = str(candidate['_id'])
    return candidate

@router.put("/{candidate_id}")
async def update_candidate(candidate_id: str, update_data: dict):
    result = candidates_collection.update_one({"_id": ObjectId(candidate_id)}, {"$set": update_data})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Candidate not found")
    candidate = candidates_collection.find_one({"_id": ObjectId(candidate_id)})
    candidate['_id'] = str(candidate['_id'])
    return candidate

@router.delete("/{candidate_id}")
async def delete_candidate(candidate_id: str):
    result = candidates_collection.delete_one({"_id": ObjectId(candidate_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return {"message": "Candidate deleted successfully"}