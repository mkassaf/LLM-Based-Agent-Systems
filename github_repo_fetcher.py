"""
github_repo_fetcher.py

Fetches repository metadata from the GitHub GraphQL API.
Requires GITHUB_TOKEN in the environment (or a .env file).
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GRAPHQL_URL = "https://api.github.com/graphql"

_QUERY = """
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    nameWithOwner
    stargazerCount
    forkCount
    primaryLanguage { name }
    licenseInfo { spdxId }
    createdAt
    updatedAt
    defaultBranchRef {
      target {
        ... on Commit {
          history { totalCount }
        }
      }
    }
    mentionableUsers { totalCount }
  }
}
"""


def fetch_repo_metadata(full_name: str) -> dict | None:
    """
    Fetch metadata for a GitHub repo given its 'owner/name' full_name.
    Returns a dict with normalized fields, or None on error.
    """
    if not GITHUB_TOKEN:
        raise EnvironmentError("GITHUB_TOKEN is not set. Check your .env file.")

    parts = full_name.strip().split("/")
    if len(parts) != 2:
        return None
    owner, name = parts

    response = requests.post(
        GRAPHQL_URL,
        json={"query": _QUERY, "variables": {"owner": owner, "name": name}},
        headers={"Authorization": f"bearer {GITHUB_TOKEN}"},
        timeout=15,
    )

    if response.status_code != 200:
        print(f"  HTTP {response.status_code} fetching {full_name}")
        return None

    body = response.json()
    if "errors" in body or "data" not in body:
        print(f"  GraphQL error for {full_name}: {body.get('errors')}")
        return None

    repo = body["data"]["repository"]
    if repo is None:
        print(f"  Repository not found: {full_name}")
        return None

    license_id = None
    if repo.get("licenseInfo"):
        license_id = repo["licenseInfo"].get("spdxId")

    commit_count = 0
    if repo.get("defaultBranchRef") and repo["defaultBranchRef"].get("target"):
        commit_count = repo["defaultBranchRef"]["target"]["history"]["totalCount"]

    return {
        "full_name":            repo["nameWithOwner"],
        "stars":                repo["stargazerCount"],
        "forks":                repo["forkCount"],
        "primary_language":     repo["primaryLanguage"]["name"] if repo["primaryLanguage"] else None,
        "license":              license_id,
        "created_at":           repo["createdAt"],
        "last_updated":         repo["updatedAt"],
        "commit_count":         commit_count,
        "number_of_contributors": repo["mentionableUsers"]["totalCount"],
    }
