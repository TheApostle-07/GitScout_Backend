from fastapi import APIRouter
from app.services.github_fetcher import (
    fetch_repos_for_user,
    fetch_commits_for_repo,
    fetch_full_user_display_data,
    fetch_github_rate_limit  # <--- import the new function
)
import openai  # Ensure you have openai installed
from app.config import settings  # Store this securely in `.env`

OPENAI_API_KEY = settings.OPENAI_API_KEY
router = APIRouter()

@router.get("/repos")
def get_repos(username: str):
    """Return all public repos for a user (raw data)."""
    repos_data, _ = fetch_repos_for_user(username)
    return repos_data

@router.get("/commits")
def get_commits(username: str, repo_name: str):
    """Return commits for a given repo (raw data)."""
    commits_data, _ = fetch_commits_for_repo(username, repo_name)
    return commits_data

@router.get("/user_full")
def get_user_full(username: str):
    """
    Combine everything:
    - user profile
    - rate-limit info -> analysis_remaining
    - total repos
    - top 3 repos by star count
    - commits for those top 3 repos
    """
    data = fetch_full_user_display_data(username)
    return data

@router.get("/rate_limit")
def get_github_rate_limit():
    """
    Return the current GitHub rate-limit usage for your token.
    """
    return fetch_github_rate_limit()

def analyze_github_profile(data):
    """
    Send GitHub profile details to GPT for structured salary estimation.
    """
    prompt = f"""
    Analyze this GitHub user's open-source contributions and estimate their salary in INR:

    - Username: {data['user_profile']['login']}
    - Name: {data['user_profile'].get('name', 'N/A')}
    - Bio: {data['user_profile'].get('bio', 'N/A')}
    - Public Repos: {data['user_profile']['public_repos']}
    - Followers: {data['user_profile']['followers']}
    - Following: {data['user_profile']['following']}
    - Total Stars on Repos: {sum(repo['stargazers_count'] for repo in data.get('top_3_repos', []))}
    - Total Forks: {sum(repo['forks_count'] for repo in data.get('top_3_repos', []))}
    - Commits in Top Repos: {sum(len(repo['commits']) for repo in data.get('top_3_repos', []))}

    Provide the response in **JSON format ONLY** without any explanation.
    Structure:
    {{
        "estimated_salary_inr": "xxxxxx - xxxxxx",
        "positives": [
            "Positive factor 1",
            "Positive factor 2",
            "Positive factor 3"
        ],
        "negatives": [
            "Negative factor 1",
            "Negative factor 2",
            "Negative factor 3"
        ],
        "improvements_needed": [
            {{
                "area": "Specific skill or metric to improve",
                "current_level": "Current status",
                "expected_level": "Expected improvement"
            }}
        ],
        "salary_after_improvements_inr": "xxxxxx - xxxxxx"
    }}
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            api_key=OPENAI_API_KEY
        )

        # Ensure GPT output is valid JSON
        gpt_output = response["choices"][0]["message"]["content"].strip()
        return eval(gpt_output)  # Convert GPT response string to JSON
    except Exception as e:
        print(f"GPT Error: {e}")
        return {
            "estimated_salary_inr": "Unavailable due to API error",
            "positives": [],
            "negatives": [],
            "improvements_needed": [],
            "salary_after_improvements_inr": "Unavailable"
        }

@router.get("/user_full_salary")
def get_user_full_salary(username: str):
    """
    Fetch complete GitHub profile details and analyze the candidate's salary.
    """
    try:
        data = fetch_full_user_display_data(username)

        # Send profile data to GPT for structured salary estimation
        gpt_analysis = analyze_github_profile(data)

        # Add AI-generated structured salary estimation
        data["salary_estimation"] = gpt_analysis

        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))