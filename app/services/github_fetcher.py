"""
github_fetcher.py

Provides functions for interacting with GitHub's API:
- Getting user profile data
- Getting repos for a user
- Getting commits for a repo
- Aggregating top-3 repos and commits for easy display
- Checking the GitHub rate limit for the current token

All functions return JSON data plus HTTP response headers for rate-limit usage if desired.
"""

import httpx
from app.config import settings

BASE_URL = "https://api.github.com"

def _get_headers():
    """
    Construct the HTTP headers, including the Bearer token from settings.
    """
    return {"Authorization": f"Bearer {settings.GITHUB_TOKEN}"}


def fetch_user_profile(username: str):
    """
    Fetch basic user profile info + parse rate-limit headers.
    Returns a tuple: (user_data, headers_dict).
    user_data: the JSON body with user info
    headers_dict: the raw HTTP response headers
    """
    url = f"{BASE_URL}/users/{username}"
    response = httpx.get(url, headers=_get_headers())
    response.raise_for_status()  # raise exception if 4xx/5xx

    user_data = response.json()
    return user_data, response.headers


def fetch_repos_for_user(username: str):
    """
    Fetch all public repositories for a given user + parse rate-limit headers.
    Returns (repos_data, headers_dict).
    repos_data: JSON list of repos
    headers_dict: raw HTTP response headers
    """
    url = f"{BASE_URL}/users/{username}/repos"
    response = httpx.get(url, headers=_get_headers())
    response.raise_for_status()
    return response.json(), response.headers


def fetch_commits_for_repo(username: str, repo_name: str):
    """
    Fetch commits for a given repo + parse rate-limit headers.
    Returns (commits_data, headers_dict).
    commits_data: JSON list of commits
    headers_dict: raw HTTP response headers

    NOTE: If a repo has many commits, consider adding '?per_page=...' to limit results.
    """
    url = f"{BASE_URL}/repos/{username}/{repo_name}/commits"
    response = httpx.get(url, headers=_get_headers())
    response.raise_for_status()
    return response.json(), response.headers


def _parse_rate_limit_info(headers):
    """
    Parse X-RateLimit-* headers from a response.
    Returns a dict with:
      - rate_limit (str)
      - rate_remaining (str)
      - rate_reset (str, epoch timestamp)
    If any are missing, returns None or default strings.

    Example usage:
      info = _parse_rate_limit_info(response.headers)
      print(info["rate_remaining"])
    """
    return {
        "rate_limit": headers.get("X-RateLimit-Limit"),
        "rate_remaining": headers.get("X-RateLimit-Remaining"),
        "rate_reset": headers.get("X-RateLimit-Reset"),
    }


def fetch_full_user_display_data(username: str):
    """
    A high-level aggregator function:

    1) Fetch the user's profile (plus rate-limit info).
    2) Fetch all repos, parse rate-limit info.
    3) Sort repos by stargazers_count (descending), pick top 3.
    4) For each top-3 repo, fetch commits (plus rate-limit info).
    5) Find the lowest "rate_remaining" among all requests, indicating how many calls remain.
    6) Return a combined JSON that includes:
       - user profile
       - overall 'analysis_remaining'
       - total repos count
       - top 3 repos + commits
    """
    # Step 1: Fetch user profile
    user_profile, user_profile_headers = fetch_user_profile(username)
    user_profile_rate_info = _parse_rate_limit_info(user_profile_headers)

    # Step 2: Fetch all user repos
    repos_data, repo_headers = fetch_repos_for_user(username)
    repos_rate_info = _parse_rate_limit_info(repo_headers)

    # Sort repos by star count desc, then pick top 3
    sorted_repos = sorted(repos_data, key=lambda r: r.get("stargazers_count", 0), reverse=True)
    top_3_repos = sorted_repos[:3]

    # Step 3: For each top-3 repo, fetch commits
    top_3_with_commits = []
    all_commits_rate_info = []

    for repo in top_3_repos:
        repo_name = repo["name"]
        commits_data, commit_headers = fetch_commits_for_repo(username, repo_name)

        commits_rate_info = _parse_rate_limit_info(commit_headers)
        all_commits_rate_info.append(commits_rate_info)

        top_3_with_commits.append({
            "repo_name": repo_name,
            "repo_url": repo.get("html_url"),
            "stargazers_count": repo.get("stargazers_count", 0),
            "forks_count": repo.get("forks_count", 0),
            "language": repo.get("language"),
            "commits": commits_data
        })

    # Step 4: Combine all rate-limit usage info
    rate_info_list = [user_profile_rate_info, repos_rate_info] + all_commits_rate_info

    remaining_values = []
    for info in rate_info_list:
        if info and info["rate_remaining"] is not None:
            try:
                remaining_values.append(int(info["rate_remaining"]))
            except ValueError:
                pass

    # The "lowest" value is effectively how many requests remain overall
    if remaining_values:
        overall_rate_remaining = min(remaining_values)
    else:
        overall_rate_remaining = None

    # Step 5: Build final result object
    combined = {
        "user_profile": {
            "login": user_profile.get("login"),
            "name": user_profile.get("name"),
            "email": user_profile.get("email"),  # can be null if private
            "avatar_url": user_profile.get("avatar_url"),
            "company": user_profile.get("company"),
            "location": user_profile.get("location"),
            "bio": user_profile.get("bio"),
            "public_repos": user_profile.get("public_repos"),
            "followers": user_profile.get("followers"),
            "following": user_profile.get("following"),
            "created_at": user_profile.get("created_at"),
            "updated_at": user_profile.get("updated_at"),
        },
        "analysis_remaining": overall_rate_remaining,  # how many more calls user can make
        "total_repos_found": len(repos_data),
        "top_3_repos": top_3_with_commits
    }

    return combined


def fetch_github_rate_limit():
    """
    Fetch the current GitHub rate-limit for the configured token
    by calling the /rate_limit endpoint.
    Returns a dict with:
      - rate_headers (parsed from X-RateLimit-*),
      - full_body (the JSON breakdown).
    """
    url = f"{BASE_URL}/rate_limit"
    response = httpx.get(url, headers=_get_headers())
    response.raise_for_status()

    # Basic rate headers (limit, remaining, reset)
    rate_headers = _parse_rate_limit_info(response.headers)

    # Detailed JSON about usage for core, search, graphql, etc.
    full_body = response.json()

    return {
        "rate_headers": rate_headers,
        "full_body": full_body
    }