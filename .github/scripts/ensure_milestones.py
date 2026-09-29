"""Create the roadmap milestones listed in .github/milestones.json and file phase-labelled issues under them.

Idempotent. Milestones that already exist (matched by title) are left as they are, so dates and descriptions edited
on GitHub are kept. Every issue labelled ``phase-N`` that has no milestone is put in the milestone whose title starts
with "Phase N:". Runs in .github/workflows/project-sync.yml with the workflow's own token (issues: write), before the
board sync, so adding a phase is a pull request that edits milestones.json plus issues with the new label.
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

TOKEN = os.environ["GITHUB_TOKEN"]
REPO = os.environ["GITHUB_REPOSITORY"]
API = os.environ.get("GITHUB_API_URL", "https://api.github.com")
MILESTONES = Path(__file__).resolve().parents[1] / "milestones.json"


def request(method, path, body=None):
    req = urllib.request.Request(
        f"{API}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={
            "Authorization": f"bearer {TOKEN}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req) as response:
        return json.load(response)


def rest_all(path):
    items, page = [], 1
    while True:
        batch = request("GET", f"{path}{'&' if '?' in path else '?'}per_page=100&page={page}")
        items += batch
        if len(batch) < 100:
            return items
        page += 1


def main():
    existing = {m["title"]: m for m in rest_all(f"/repos/{REPO}/milestones?state=all")}
    for wanted in json.loads(MILESTONES.read_text(encoding="utf-8")):
        if wanted["title"] in existing:
            continue
        created = request(
            "POST",
            f"/repos/{REPO}/milestones",
            {
                "title": wanted["title"],
                "due_on": f"{wanted['due_on']}T00:00:00Z",
                "description": f"{wanted['description']} (target date is tentative)",
            },
        )
        existing[created["title"]] = created
        print(f"Created milestone {created['title']}")

    phases = {}
    for milestone in existing.values():
        match = re.match(r"Phase (\d+):", milestone["title"])
        if match:
            phases[int(match.group(1))] = milestone
    for issue in rest_all(f"/repos/{REPO}/issues?state=all"):
        if "pull_request" in issue or issue.get("milestone"):
            continue
        numbers = [int(m[1]) for m in (re.fullmatch(r"phase-(\d+)", label["name"]) for label in issue["labels"]) if m]
        if len(numbers) == 1 and numbers[0] in phases:
            milestone = phases[numbers[0]]
            request("PATCH", f"/repos/{REPO}/issues/{issue['number']}", {"milestone": milestone["number"]})
            print(f"#{issue['number']} -> {milestone['title']}")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as error:
        print(f"::error::{error.code} {error.read().decode()[:500]}")
        sys.exit(1)
