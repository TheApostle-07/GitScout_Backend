from fastapi import APIRouter, HTTPException, Body, UploadFile, File, Query
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from bson import ObjectId
import numpy as np
import pandas as pd
from numpy.linalg import norm
from typing import Dict, List, Optional
from pydantic import BaseModel

from app.config import settings
from app.services.embedding import (
    generate_ideal_candidate_vector,
    generate_candidate_vector
)
from app.services.github_fetcher import fetch_full_user_display_data

import openai
import os

router = APIRouter()

# Initialize MongoDB client for the jobs collection
client = MongoClient(settings.MONGO_URI)
db = client[settings.DATABASE_NAME]
jobs_collection = db["jobs"]

# -------------------------------
#  Job CRUD Endpoints
# -------------------------------
@router.post("/")
async def create_job(job: dict = Body(...)):
    """
    Create a new job posting.
    The job document should include ideal candidate criteria under 'idealMetrics'.
    This function generates and stores an 'ideal_vector' based on those metrics.
    """
    criteria: Dict = job.get("idealMetrics")
    if not criteria:
        raise HTTPException(
            status_code=400,
            detail="Job must include 'idealMetrics' to generate ideal vector."
        )
    
    ideal_vector = generate_ideal_candidate_vector(criteria)
    ideal_vector = np.concatenate((ideal_vector, np.array([0.0], dtype=float)))
    ideal_vector = ideal_vector / norm(ideal_vector)
    job["ideal_vector"] = ideal_vector.tolist()

    result = jobs_collection.insert_one(job)
    job["_id"] = str(result.inserted_id)
    return job

@router.get("/")
async def get_all_jobs():
    jobs = list(jobs_collection.find())
    for job in jobs:
        job['_id'] = str(job['_id'])
    return jobs

@router.get("/{job_id}")
async def get_job(job_id: str):
    job = jobs_collection.find_one({"_id": ObjectId(job_id)})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job['_id'] = str(job['_id'])
    return job

@router.put("/{job_id}")
async def update_job(job_id: str, update_data: dict = Body(...)):
    """
    Update an existing job. If 'ideal_candidate_criteria' is updated, regenerate the 'ideal_vector'.
    """
    if "ideal_candidate_criteria" in update_data:
        criteria = update_data["ideal_candidate_criteria"]
        new_vector = generate_ideal_candidate_vector(criteria)
        new_vector = np.concatenate((new_vector, np.array([0.0], dtype=float)))
        new_vector = new_vector / norm(new_vector)
        update_data["ideal_vector"] = new_vector.tolist()

    result = jobs_collection.update_one({"_id": ObjectId(job_id)}, {"$set": update_data})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_collection.find_one({"_id": ObjectId(job_id)})
    job['_id'] = str(job['_id'])
    return job

@router.delete("/{job_id}")
async def delete_job(job_id: str):
    result = jobs_collection.delete_one({"_id": ObjectId(job_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"message": "Job deleted successfully"}

# -------------------------------
#  Candidate Matching Endpoints
# -------------------------------

# Pydantic models for candidate input
class CandidateInput(BaseModel):
    github_username: str
    user_profile: Optional[dict] = None
    top_3_repos: Optional[List[dict]] = None

class CandidatesPayload(BaseModel):
    candidates: List[CandidateInput]

@router.get("/{job_id}/match-single-candidate")
async def match_single_candidate(
    job_id: str,
    github_username: str = Query(..., description="GitHub username to fetch candidate data")
):
    """
    Matches a single candidate by fetching GitHub data, generating the candidate vector,
    and comparing it to the job's stored ideal_vector using cosine similarity.
    Returns a decision of "Hire" or "No-Hire".
    """
    job = jobs_collection.find_one({"_id": ObjectId(job_id)})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if "ideal_vector" not in job:
        raise HTTPException(status_code=400, detail="This job does not contain a stored 'ideal_vector'.")

    ideal_metrics = job.get("idealMetrics", {})
    ideal_vector = np.array(job["ideal_vector"], dtype=np.float32)
    ideal_vector /= norm(ideal_vector)

    try:
        candidate_data = fetch_full_user_display_data(github_username)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error fetching data for GitHub user '{github_username}': {str(e)}"
        )

    try:
        cand_vector = generate_candidate_vector(candidate_data, ideal_metrics)
        cand_vector /= norm(cand_vector)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error generating vector for GitHub user '{github_username}': {str(e)}"
        )

    similarity_score = float(np.dot(ideal_vector, cand_vector))
    hire_decision = "Hire" if similarity_score >= settings.SIMILARITY_THRESHOLD else "No-Hire"

    return {
        "github_username": github_username,
        "similarity_score": similarity_score,
        "decision": hire_decision
    }

@router.post("/{job_id}/batch-match")
async def batch_match_job_candidates(
    job_id: str,
    file: UploadFile = File(...)
):
    """
    Processes a CSV file of candidate records for a specific job.
    Each candidate is evaluated by generating a vector and comparing it to the job's ideal_vector.
    """
    job = jobs_collection.find_one({"_id": ObjectId(job_id)})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if "ideal_vector" not in job:
        raise HTTPException(status_code=400, detail="Job does not contain a stored 'ideal_vector'.")

    ideal_vector = np.array(job["ideal_vector"], dtype=np.float32)
    ideal_vector /= norm(ideal_vector)

    try:
        contents = await file.read()
        from io import StringIO
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {e}")

    candidates_data = df.to_dict(orient="records")
    results = []
    similarity_threshold = settings.SIMILARITY_THRESHOLD

    for candidate in candidates_data:
        try:
            cand_vector = generate_candidate_vector(candidate, job.get("idealMetrics", {}))
            cand_vector = np.concatenate((cand_vector, np.array([0.0], dtype=float)))
            cand_vector = cand_vector / norm(cand_vector)
            sim = float(np.dot(ideal_vector, cand_vector))
            hire_decision = "Hire" if sim >= similarity_threshold else "No-Hire"
            results.append({
                "candidate": candidate,
                "similarity_score": sim,
                "hire_decision": hire_decision
            })
        except Exception as e:
            results.append({
                "candidate": candidate,
                "error": f"Error generating vector: {str(e)}"
            })
    
    return JSONResponse(content={"results": results})

from pydantic import BaseModel
from typing import List, Dict

class CandidateData(BaseModel):
    github_username: str
    other_field: str  # Modify based on your expected fields

class CandidateRequest(BaseModel):
    candidates: List[CandidateData]

@router.post("/{job_id}/match-multiple-candidates")
async def match_multiple_candidates(
    job_id: str,
    request: CandidateRequest = Body(...)
):
    """
    Matches a list of candidate profiles (sent as JSON) against this job's stored ideal_vector.
    """
    job = jobs_collection.find_one({"_id": ObjectId(job_id)})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if "ideal_vector" not in job:
        raise HTTPException(status_code=400, detail="Job does not contain a stored 'ideal_vector'.")

    candidates = request.candidates  # Extract candidates list
    ideal_metrics = job.get("idealMetrics", {})
    ideal_vector = np.array(job["ideal_vector"], dtype=np.float32)
    ideal_vector /= norm(ideal_vector)
    expected_dim = ideal_vector.shape[0]  # Corrected indexing

    results = []
    threshold = settings.SIMILARITY_THRESHOLD

    for candidate_data in candidates:
        try:
            # Fetch full data if missing
            if not candidate_data.get("user_profile") or not candidate_data.get("top_3_repos"):
                candidate_data = fetch_full_user_display_data(candidate_data.github_username)

            # Generate candidate vector
            cand_vec = generate_candidate_vector(candidate_data, ideal_metrics)
            
            # Ensure proper dimension alignment
            if cand_vec.shape[0] == expected_dim - 1:
                cand_vec = np.concatenate((cand_vec, np.array([0.0], dtype=float)))
            elif cand_vec.shape[0] != expected_dim:
                raise Exception(f"Candidate vector shape {cand_vec.shape} does not match expected {(expected_dim,)}")

            # Normalize candidate vector
            cand_vec = cand_vec / norm(cand_vec)
            similarity = float(np.dot(ideal_vector, cand_vec))
            hire_decision = "Hire" if similarity >= threshold else "No-Hire"

            results.append({
                "candidate": candidate_data.dict(),
                "similarity_score": similarity,
                "decision": hire_decision
            })
        except Exception as e:
            results.append({
                "candidate": candidate_data.dict(),
                "error": f"Error generating vector: {str(e)}"
            })

    return {"results": results}

# -------------------------------
#  AI Commit Message Generation Endpoint
# -------------------------------
class JobDetailsForCommitGeneration(BaseModel):
    title: str
    description: str

openai.api_key = settings.OPENAI_API_KEY

@router.post("/generate-commit-messages")
async def generate_commit_messages(details: JobDetailsForCommitGeneration = Body(...)):
    prompt = (
        f"You are an expert software engineer and technical recruiter. "
        f"Given the job title '{details.title}' and the job description '{details.description}', "
        f"generate a list of 10 concise and relevant commit messages that would resonate with "
        f"an ideal candidate’s work history. Each commit message should clearly reflect a meaningful contribution."
    )
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        raw_output = response.choices[0].message['content']
        commit_messages = [
            line.strip().lstrip("-").lstrip("0123456789.").strip()
            for line in raw_output.split("\n")
            if line.strip()
        ]
        commit_messages = commit_messages[:10]
        
        return {"commit_messages": commit_messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))