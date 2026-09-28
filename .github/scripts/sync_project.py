"""Create and sync the "EtAlii.Dllm" GitHub Project (v2) from this repository's issues and milestones.

Idempotent: every run converges the board to the repository state, so it can run on any event or schedule.
- Creates the user-owned project if it does not exist and links it to the repository.
- Ensures the fields "Phase" (single select), "Start date" and "Target date" exist.
- Adds every issue to the project. "Phase" comes from the issue's milestone ("Phase N: ..."), "Target date" from
  the milestone's due date and "Start date" from the previous phase's due date.
- Sets the built-in "Status" to Done for closed issues and to Todo for open issues that have no status yet;
  other manual status changes are left alone.
- Tries to create a "Roadmap" view (only possible where the API supports it).

Needs PROJECT_TOKEN: a classic token with only the `project` scope. Fine-grained tokens cannot reach user-owned
Projects. The repository is public, so reading its issues needs no further scope.
"""

import datetime
import json
import os
import re
import sys
import urllib.error
import urllib.request

TOKEN = os.environ["PROJECT_TOKEN"]
REPO = os.environ["GITHUB_REPOSITORY"]
OWNER, NAME = REPO.split("/")
TITLE = os.environ.get("PROJECT_TITLE", "EtAlii.Dllm")
API = os.environ.get("GITHUB_API_URL", "https://api.github.com")
PHASE_COUNT = 7
COLORS = ["GREEN", "BLUE", "PURPLE", "GRAY", "YELLOW", "ORANGE", "RED"]


def request(method, url, body=None, accept="application/vnd.github+json"):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"bearer {TOKEN}",
        "Accept": accept,
        "Content-Type": "application/json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    with urllib.request.urlopen(req) as response:
        return json.load(response)


def graphql(query, **variables):
    result = request("POST", f"{API}/graphql", {"query": query, "variables": variables})
    if result.get("errors"):
        raise RuntimeError(json.dumps(result["errors"], indent=2))
    return result["data"]


def rest_all(path):
    items, page = [], 1
    while True:
        batch = request("GET", f"{API}{path}{'&' if '?' in path else '?'}per_page=100&page={page}")
        items += batch
        if len(batch) < 100:
            return items
        page += 1


def ensure_project():
    data = graphql("""query($owner: String!, $name: String!) {
        user(login: $owner) { id projectsV2(first: 100) { nodes { id number title url repositories(first: 20) { nodes { id } } } } }
        repository(owner: $owner, name: $name) { id }
    }""", owner=OWNER, name=NAME)
    repo_id = data["repository"]["id"]
    project = next((p for p in data["user"]["projectsV2"]["nodes"] if p["title"] == TITLE), None)
    if project is None:
        project = graphql("""mutation($owner: ID!, $title: String!) {
            createProjectV2(input: {ownerId: $owner, title: $title}) { projectV2 { id number title url } }
        }""", owner=data["user"]["id"], title=TITLE)["createProjectV2"]["projectV2"]
        print(f"Created project {project['url']}")
        graphql("""mutation($id: ID!, $desc: String!, $readme: String!) {
            updateProjectV2(input: {projectId: $id, shortDescription: $desc, readme: $readme}) { projectV2 { id } }
        }""", id=project["id"], desc="Roadmap and progress of the deterministic LLM",
                readme=f"Synced automatically from https://github.com/{REPO} issues and milestones by "
                       "`.github/workflows/project-sync.yml`. Edit issues and milestones there, not here.")
    if any(r["id"] == repo_id for r in project.get("repositories", {}).get("nodes", [])):
        return project
    try:
        graphql("""mutation($p: ID!, $r: ID!) {
            linkProjectV2ToRepository(input: {projectId: $p, repositoryId: $r}) { repository { id } }
        }""", p=project["id"], r=repo_id)
    except RuntimeError as error:  # already linked, or the token may not link repositories
        print(f"::notice::Project not linked to the repository ({error.args[0][:200]}). "
              "Link it once by hand: project > ... > Settings > Linked repositories.")
    return project


FIELDS_QUERY = """query($id: ID!) { node(id: $id) { ... on ProjectV2 { fields(first: 50) { nodes {
    ... on ProjectV2FieldCommon { id name dataType }
    ... on ProjectV2SingleSelectField { options { id name } } } } } } }"""


def ensure_fields(project, phase_names):
    fields = {f["name"]: f for f in graphql(FIELDS_QUERY, id=project["id"])["node"]["fields"]["nodes"] if f}
    if "Phase" not in fields:
        options = [{"name": n, "color": COLORS[i % len(COLORS)], "description": ""} for i, n in enumerate(phase_names)]
        graphql("""mutation($p: ID!, $o: [ProjectV2SingleSelectFieldOptionInput!]) {
            createProjectV2Field(input: {projectId: $p, dataType: SINGLE_SELECT, name: "Phase",
                                         singleSelectOptions: $o}) { projectV2Field { __typename } } }""",
                p=project["id"], o=options)
    for name in ("Start date", "Target date"):
        if name not in fields:
            graphql("""mutation($p: ID!, $n: String!) {
                createProjectV2Field(input: {projectId: $p, dataType: DATE, name: $n}) { projectV2Field { __typename } }
            }""", p=project["id"], n=name)
    fields = {f["name"]: f for f in graphql(FIELDS_QUERY, id=project["id"])["node"]["fields"]["nodes"] if f}
    missing = [n for n in phase_names if n not in {o["name"] for o in fields["Phase"]["options"]}]
    if missing:
        print(f"::warning::Add these options to the Phase field by hand: {', '.join(missing)}")
    return fields


def existing_items(project):
    items, cursor = {}, None
    while True:
        page = graphql("""query($id: ID!, $after: String) { node(id: $id) { ... on ProjectV2 {
            items(first: 100, after: $after) { pageInfo { hasNextPage endCursor } nodes { id
                content { ... on Issue { id } ... on PullRequest { id } }
                status: fieldValueByName(name: "Status") { ... on ProjectV2ItemFieldSingleSelectValue { name } }
            } } } } }""", id=project["id"], after=cursor)["node"]["items"]
        for node in page["nodes"]:
            if node["content"]:
                items[node["content"]["id"]] = node
        if not page["pageInfo"]["hasNextPage"]:
            return items
        cursor = page["pageInfo"]["endCursor"]


def set_value(project, item_id, field, value):
    graphql("""mutation($p: ID!, $i: ID!, $f: ID!, $v: ProjectV2FieldValue!) {
        updateProjectV2ItemFieldValue(input: {projectId: $p, itemId: $i, fieldId: $f, value: $v}) {
            projectV2Item { id } } }""", p=project["id"], i=item_id, f=field["id"], v=value)


def option(field, name):
    return next((o["id"] for o in field.get("options", []) if o["name"].lower() == name.lower()), None)


def ensure_roadmap_view(project):
    url = f"{API}/users/{OWNER}/projectsV2/{project['number']}/views"
    try:
        views = request("GET", url)
        if any(v.get("layout") == "roadmap" for v in views):
            return
        request("POST", url, {"name": "Roadmap", "layout": "roadmap"})
        print("Created Roadmap view")
    except urllib.error.HTTPError as error:
        print(f"::notice::Could not create the Roadmap view through the API ({error.code}). Add it once by hand: "
              f"{project['url']} > New view > Roadmap, dates 'Start date' to 'Target date', group by Phase.")


def main():
    milestones = rest_all(f"/repos/{REPO}/milestones?state=all")
    phases = {}
    for milestone in milestones:
        match = re.match(r"Phase (\d+)", milestone["title"])
        if match:
            phases[int(match.group(1))] = milestone
    phase_names = [phases[i]["title"] if i in phases else f"Phase {i}" for i in range(max(PHASE_COUNT, len(phases)))]

    project = ensure_project()
    fields = ensure_fields(project, phase_names)
    items = existing_items(project)

    issues = [i for i in rest_all(f"/repos/{REPO}/issues?state=all") if "pull_request" not in i]
    for issue in sorted(issues, key=lambda i: i["number"]):
        item = items.get(issue["node_id"])
        if item is None:
            item_id = graphql("""mutation($p: ID!, $c: ID!) {
                addProjectV2ItemById(input: {projectId: $p, contentId: $c}) { item { id } } }""",
                              p=project["id"], c=issue["node_id"])["addProjectV2ItemById"]["item"]["id"]
            status = None
        else:
            item_id, status = item["id"], (item["status"] or {}).get("name")

        milestone = issue.get("milestone")
        match = re.match(r"Phase (\d+)", milestone["title"]) if milestone else None
        if match:
            number = int(match.group(1))
            option_id = option(fields["Phase"], milestone["title"])
            if option_id:
                set_value(project, item_id, fields["Phase"], {"singleSelectOptionId": option_id})
            if milestone.get("due_on"):
                set_value(project, item_id, fields["Target date"], {"date": milestone["due_on"][:10]})
            previous = phases.get(number - 1)
            if previous and previous.get("due_on"):
                start = datetime.date.fromisoformat(previous["due_on"][:10]) + datetime.timedelta(days=1)
            else:
                start = datetime.date.fromisoformat(milestone["created_at"][:10])
            set_value(project, item_id, fields["Start date"], {"date": start.isoformat()})

        status_field = fields.get("Status")
        if status_field:
            wanted = "Done" if issue["state"] == "closed" else ("Todo" if status is None else None)
            if wanted and wanted != status and option(status_field, wanted):
                set_value(project, item_id, status_field, {"singleSelectOptionId": option(status_field, wanted)})
        print(f"#{issue['number']} {issue['title']}")

    ensure_roadmap_view(project)
    print(f"Project: {project['url']}")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as error:
        print(f"::error::{error.code} {error.read().decode()[:500]}")
        sys.exit(1)
