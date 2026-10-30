# The roadmap's GitHub issues

`sync_issues.py` mirrors the milestones of `docs/helmholtz/roadmap.md` (HR, H0 to H8) as
GitHub issues, one per milestone, labelled `enhancement` and `helmholtz`:

- the body holds the milestone's gate, its progress, and every task as a task-list item,
  ticked as the roadmap ticks it, with the *Done when*, *Needs* and the merged pull requests
  titled `Helmholtz <ID>: …`;
- the issue is closed when the milestone's gate (`<milestone>.G`) is ticked, and reopened if
  it is unticked;
- an issue is found by the marker `<!-- roadmap-sync: <milestone> -->` in its body, so a
  renamed milestone keeps its issue.

The roadmap is the source of truth. An edit made in an issue, a ticked box included, is
overwritten by the next sync; change the roadmap instead. A split task, a reworded *Done
when* or a reordered milestone reaches its issue with the next push to `main`.

The workflow `.github/workflows/roadmap-issues.yml` runs `sync_issues.py --apply` on every
push to `main` that changes the roadmap or this directory, and by hand
(`gh workflow run roadmap-issues.yml`). A task's box therefore ticks when its PR is merged,
not when its branch ticks it. By hand, with `gh` logged in:

```sh
python3 tools/roadmap/sync_issues.py                    # dry run: what would change
python3 tools/roadmap/sync_issues.py --out <dir>        # dry run, bodies written to <dir>
python3 tools/roadmap/sync_issues.py --check            # dry run, exit 1 on any drift
python3 tools/roadmap/sync_issues.py --apply            # create, edit, close, reopen
python3 tools/roadmap/sync_issues.py --issue-of CH1.2c  # the task's milestone issue
```

`/next-task` and `/milestone-gate` put `Part of #<n>` in their PR text, with `<n>` from
`--issue-of`, so a task's PR shows in its milestone's issue. The script needs Python 3
and `gh`, nothing else.
