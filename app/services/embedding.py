import openai
import numpy as np
from typing import Dict, List
from app.config import settings

# Set your OpenAI API key from settings
openai.api_key = settings.OPENAI_API_KEY

def get_openai_embedding(text: str) -> np.ndarray:
    """
    Calls the OpenAI API to generate an embedding using text-embedding-ada-002.
    Returns the embedding as a NumPy array.
    """
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    # Extract the embedding from the first result in the response.
    embedding = response["data"][0]["embedding"]
    return np.array(embedding, dtype=np.float32)

def generate_candidate_embeddings(candidates: List[Dict], ideal_metrics: Dict) -> np.ndarray:
    """
    Given a list of candidate documents and the ideal metrics, generates an embedding for each candidate.
    Each candidate embedding is generated using the generate_candidate_vector function.
    Returns a NumPy array of embeddings.
    """
    embeddings = []
    for candidate in candidates:
        embedding = generate_candidate_vector(candidate, ideal_metrics)
        embeddings.append(embedding)
    return np.array(embeddings)

def generate_ideal_candidate_vector(request_data: Dict) -> np.ndarray:
    """
    Converts the ideal candidate request data (e.g. languages, min_commits) into a standardized textual
    description and generates its embedding using OpenAI's text-embedding-ada-002.
    
    Expects request_data as a dictionary with keys such as "languages" and "min_commits".
    """
    text_parts = []
    languages = request_data.get("languages", [])
    if languages:
        text_parts.append("Languages: " + ", ".join(languages))
    min_commits = request_data.get("min_commits")
    if min_commits:
        text_parts.append("Min commits: " + str(min_commits))
    # Additional ideal criteria can be added here.
    combined_text = " || ".join(text_parts).strip()
    if not combined_text:
        combined_text = "No ideal candidate criteria provided."
    
    return get_openai_embedding(combined_text)

def generate_candidate_vector(candidate_data: Dict, ideal_metrics: Dict) -> np.ndarray:
    """
    Generates an embedding for a candidate using the provided GitHub data structure,
    but only if the candidate’s key factors can be determined.
    
    Expected candidate_data structure:
      - 'user_profile': a dict containing fields like 'login', 'name', 'bio', 'location', 'company'
      - 'top_3_repos': a list of repositories with keys like:
            'repo_name', 'repo_url', 'stargazers_count', 'language', and 'commits'
        Each 'commits' is a list of commit objects with a nested 'commit' dict that includes:
            { "message": "..." }
    
    The function derives candidate factors:
      1. Commit Frequency: total commit count mapped to qualitative levels.
      2. Code Readability: average length of commit messages mapped to a level.
      3. Technology Match: aggregated languages from the candidate’s repositories.
    
    It then creates a combined text that includes repository details, profile information,
    and a summary of candidate factors alongside the provided ideal metrics.
    
    A text embedding is generated from this text, and a numeric feature (total stars)
    is concatenated to produce the final candidate vector.
    
    Raises an Exception if necessary candidate factors are missing.
    """
    text_segments = []
    
    # Process repository information.
    repos = candidate_data.get("top_3_repos", [])
    
    # 1. Derive Commit Frequency.
    total_commits = sum(len(repo.get("commits", [])) for repo in repos)
    if total_commits < 5:
        candidate_commit_frequency = "Low"
    elif total_commits < 10:
        candidate_commit_frequency = "Moderate (2-3 commits/day)"
    else:
        candidate_commit_frequency = "High"
    
    # 2. Derive Code Readability from commit messages.
    all_commit_messages = []
    for repo in repos:
        for commit_obj in repo.get("commits", []):
            msg = commit_obj.get("commit", {}).get("message", "")
            if msg:
                all_commit_messages.append(msg)
    if all_commit_messages:
        avg_length = sum(len(msg) for msg in all_commit_messages) / len(all_commit_messages)
        candidate_code_readability = "Very High" if avg_length > 50 else "Moderate"
    else:
        candidate_code_readability = "Unknown"
    
    # 3. Derive Technology Match from repository languages.
    candidate_languages = {repo.get("language") for repo in repos if repo.get("language")}
    candidate_match = ", ".join(candidate_languages) if candidate_languages else "None"
    
    # Validate that the candidate factors are properly determined.
    if candidate_code_readability == "Unknown" or candidate_match == "None":
        raise Exception("Candidate factors not properly determined; cannot generate vector.")
    
    # Build a candidate factors summary (including ideal metrics for context).
    candidate_factors_summary = (
        f"Candidate Commit Frequency: {candidate_commit_frequency} || "
        f"Candidate Code Readability: {candidate_code_readability} || "
        f"Candidate Languages: {candidate_match} || "
        f"Ideal Commit Frequency: {ideal_metrics.get('commitFrequency', 'Not Provided')} || "
        f"Ideal Code Readability: {ideal_metrics.get('codeReadability', 'Not Provided')} || "
        f"Ideal Match: {ideal_metrics.get('matchToJobRequirements', 'Not Provided')}"
    )
    
    # Build detailed text for embedding.
    for repo in repos:
        repo_name = repo.get("repo_name", "")
        language = repo.get("language", "")
        text_segments.append(f"Repo: {repo_name}, Language: {language}")
        for commit_obj in repo.get("commits", []):
            commit_message = commit_obj.get("commit", {}).get("message", "")
            if commit_message:
                text_segments.append(f"Commit message: {commit_message}")
    
    # Include candidate profile details.
    user_profile = candidate_data.get("user_profile", {})
    for key in ["login", "name", "bio", "location", "company"]:
        value = user_profile.get(key)
        if value:
            text_segments.append(f"{key.capitalize()}: {value}")
    
    # Append the candidate factors summary.
    text_segments.append(candidate_factors_summary)
    
    combined_text = " || ".join(text_segments).strip()
    if not combined_text:
        combined_text = "No repository or profile information available."
    
    print("Candidate combined text:", combined_text)  # Debug output
    
    # Generate text embedding.
    text_vector = get_openai_embedding(combined_text)
    
    # Calculate numeric feature: sum of stars from repositories.
    total_stars = sum(repo.get("stargazers_count", 0) for repo in repos)
    numeric_vector = np.array([total_stars], dtype=float)
    
    # Final candidate vector is the concatenation of the text embedding and the numeric feature.
    candidate_vector = np.concatenate((text_vector, numeric_vector))
    return candidate_vector