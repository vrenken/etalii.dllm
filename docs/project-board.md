# Project board

Progress is tracked in the user-owned GitHub Project **EtAlii.Dllm**, linked to this repository.

- **Source of truth: issues and milestones in this repository.** Each roadmap phase in the README is a milestone
  (`Phase N: ...`, with a tentative due date) and each roadmap item is an issue labelled `roadmap` and `phase-N`.
- `.github/workflows/project-sync.yml` runs `.github/scripts/sync_project.py` on issue and milestone changes, daily,
  and on demand. It creates the project if needed, adds every issue, and sets `Phase`, `Start date`,
  `Target date` and `Status` (Done when closed). Edit issues and milestones, not the board.
- The **Roadmap** view shows items from `Start date` to `Target date`, grouped by `Phase`.

## One-time setup

GitHub's Actions token cannot write to a user's Projects, so the sync needs a personal token:

1. Create a classic token at <https://github.com/settings/tokens/new> with **only the `project` scope** ticked
   and an expiration of your choice (renew the secret when it expires). Fine-grained tokens have no permission
   for user-owned Projects, so a classic token is required. Because the repository is public, the token needs no
   repository scope: it can edit your Projects and read public data, but cannot push or change issues.
2. Add it as the repository secret `PROJECT_TOKEN`
   (Settings > Secrets and variables > Actions > New repository secret).
3. Run the *Project board sync* workflow once (Actions > Project board sync > Run workflow).
4. If the run log says the project could not be linked to the repository, link it once by hand
   (project > Settings > Linked repositories).
5. If the run log says the Roadmap view could not be created through the API, add it once by hand in the
   project: New view > Roadmap, dates `Start date` / `Target date`, group by `Phase`.

## Keeping it current (for Claude sessions)

When a PR finishes a roadmap item, put `Closes #<issue>` in its description. When the README roadmap changes, add,
retitle or close the matching issues and milestones in the same PR's session. A scheduled Claude routine also
reconciles the roadmap, merged PRs and issues periodically.
