#!/usr/bin/env python3
"""Mirror the milestones of docs/helmholtz/roadmap.md as GitHub issues.

One issue per milestone (HR, H0 to H8), labelled `enhancement` and `helmholtz`. Its body
holds the milestone's gate and every task of the roadmap as a task-list item, ticked as
the roadmap ticks it, with the merged pull requests titled `Helmholtz <ID>: …` linked.
The issue is closed when the milestone's gate (`<milestone>.G`) is ticked, and reopened
if it is unticked. An issue is found by the marker `<!-- roadmap-sync: <milestone> -->`
in its body, so a renamed title creates no duplicate. The roadmap is the source of truth:
edits made in an issue are overwritten by the next sync.

The workflow `.github/workflows/roadmap-issues.yml` runs it with `--apply` on every push
to `main` that changes the roadmap. Standard library and the `gh` CLI only.

    python3 tools/roadmap/sync_issues.py                # dry run: print what would change
    python3 tools/roadmap/sync_issues.py --out DIR      # dry run, bodies written to DIR
    python3 tools/roadmap/sync_issues.py --check        # dry run, exit 1 on any drift
    python3 tools/roadmap/sync_issues.py --apply        # create, edit, close, reopen
    python3 tools/roadmap/sync_issues.py --issue-of CH1.2c   # the task's milestone issue
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROADMAP = "docs/helmholtz/roadmap.md"
LABELS = {
    "enhancement": ("a2eeef", "New feature or request"),
    "helmholtz": ("5319e7", "Helmholtz FMM roadmap (docs/helmholtz/roadmap.md)"),
}
MARKER = "<!-- roadmap-sync: {} -->"
MILESTONE = re.compile(r"^## (H[R0-9]): (.+)$")
ITEM = re.compile(r"^- (?:\[( |x)\] )?(.*)$")
TASK_ID = re.compile(r"^\**((?:C?H[R0-9])(?:\.[0-9A-Za-z]+)+)\b")


def parse(text):
    """Milestones in file order: id, title, heading, gate, items (id, done, lines)."""
    milestones = []
    current = None
    item = None
    in_gate = False
    for line in text.splitlines():
        m = MILESTONE.match(line)
        if m:
            current = {"id": m[1], "title": m[2], "heading": line[3:], "gate": [],
                       "items": []}
            milestones.append(current)
            item = None
            continue
        if current is None:
            continue
        if line.startswith("## "):
            current = item = None
            continue
        if line.startswith("**Gate:**") or (in_gate and line.strip()):
            in_gate = True
            current["gate"].append(line)
            continue
        in_gate = False
        m = ITEM.match(line)
        if m:
            t = TASK_ID.match(m[2])
            if t is None:
                sys.exit(f"{ROADMAP}: no task ID in {line!r}")
            item = {"id": t[1], "done": None if m[1] is None else m[1] == "x",
                    "lines": [m[2]]}
            current["items"].append(item)
        elif item is not None and line.startswith("  ") and line.strip():
            item["lines"].append(line.strip())
        elif not line.strip():
            item = None
    ids = [i["id"] for m in milestones for i in m["items"]]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        sys.exit(f"{ROADMAP}: task IDs repeated: {', '.join(dupes)}")
    for m in milestones:
        if not m["items"] or m["items"][-1]["id"] != f"{m['id']}.G":
            sys.exit(f"{ROADMAP}: milestone {m['id']} does not end with {m['id']}.G")
    return milestones


def slug(heading):
    """GitHub's anchor for a Markdown heading."""
    return re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")


def gh(*args, stdin=None):
    out = subprocess.run(["gh", *args], cwd=ROOT, input=stdin, text=True,
                         capture_output=True)
    if out.returncode != 0:
        sys.exit(f"gh {' '.join(args)} failed:\n{out.stderr}")
    return out.stdout


def merged_prs():
    """Task ID -> numbers of merged PRs titled `Helmholtz <ID>: …`, oldest first."""
    prs = json.loads(gh("pr", "list", "--state", "merged", "--limit", "1000",
                        "--search", "Helmholtz in:title",
                        "--json", "number,title"))
    by_id = {}
    for pr in sorted(prs, key=lambda p: p["number"]):
        m = re.match(r"^Helmholtz (\S+?):", pr["title"])
        if m:
            by_id.setdefault(m[1], []).append(pr["number"])
    return by_id


def body(milestone, repo, prs):
    url = f"https://github.com/{repo}/blob/main/{ROADMAP}#{slug(milestone['heading'])}"
    tasks = [i for i in milestone["items"] if i["done"] is not None]
    done = sum(i["done"] for i in tasks)
    out = [
        MARKER.format(milestone["id"]),
        f"> Mirrored from [`{ROADMAP}`]({url}) on `main` by `tools/roadmap/sync_issues.py`"
        " after every change to it; edits made here are overwritten. Change the roadmap"
        " instead.",
        "",
        " ".join(milestone["gate"]) or "**Gate:** –",
        "",
        f"**Progress:** {done} of {len(tasks)} done.",
        "",
        "### Tasks",
        "",
    ]
    for i in milestone["items"]:
        box = "" if i["done"] is None else ("[x] " if i["done"] else "[ ] ")
        text = " ".join(i["lines"])
        numbers = prs.get(i["id"], [])
        if numbers:
            text += " Merged: " + ", ".join(f"#{n}" for n in numbers) + "."
        out.append(f"- {box}{text}")
    return "\n".join(out) + "\n"


def title(milestone):
    return f"Helmholtz {milestone['id']}: {milestone['title']}"


def existing_issues():
    """Milestone ID -> issue (number, title, body, state) for every marked issue."""
    issues = json.loads(gh("issue", "list", "--state", "all", "--limit", "1000",
                           "--search", "roadmap-sync in:body",
                           "--json", "number,title,body,state"))
    found = {}
    for issue in issues:
        m = re.search(r"<!-- roadmap-sync: (H[R0-9]) -->", issue["body"] or "")
        if m:
            if m[1] in found:
                sys.exit(f"two issues carry the marker of {m[1]}: "
                         f"#{found[m[1]]['number']} and #{issue['number']}")
            found[m[1]] = issue
    return found


def ensure_labels():
    have = {l["name"] for l in json.loads(gh("label", "list", "--limit", "200",
                                             "--json", "name"))}
    for name, (color, description) in LABELS.items():
        if name not in have:
            gh("label", "create", name, "--color", color, "--description", description)


def main():
    p = argparse.ArgumentParser(description="Mirror the roadmap's milestones as GitHub issues.")
    p.add_argument("--apply", action="store_true", help="make the changes")
    p.add_argument("--check", action="store_true", help="exit 1 if anything would change")
    p.add_argument("--out", type=pathlib.Path, help="write the issue bodies to this directory")
    p.add_argument("--issue-of", metavar="ID",
                   help="print the issue number of the milestone holding this task")
    p.add_argument("--repo", help="owner/name (default: gh's repository here)")
    args = p.parse_args()

    milestones = parse((ROOT / ROADMAP).read_text())
    issues = existing_issues()

    if args.issue_of:
        for m in milestones:
            if args.issue_of in (i["id"] for i in m["items"]) or args.issue_of == m["id"]:
                if m["id"] not in issues:
                    sys.exit(f"{m['id']} has no issue yet")
                print(issues[m["id"]]["number"])
                return
        sys.exit(f"{args.issue_of} is not in {ROADMAP}")

    repo = args.repo or json.loads(gh("repo", "view", "--json", "nameWithOwner"))[
        "nameWithOwner"]
    prs = merged_prs()
    if args.apply:
        ensure_labels()
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    drift = False
    for m in milestones:
        text = body(m, repo, prs)
        if args.out:
            (args.out / f"{m['id']}.md").write_text(f"# {title(m)}\n\n{text}")
        closed = m["items"][-1]["done"]
        issue = issues.get(m["id"])
        actions = []
        if issue is None:
            actions.append("create")
            if closed:
                actions.append("close")
        else:
            if issue["title"] != title(m) or issue["body"].strip() != text.strip():
                actions.append("edit")
            if closed and issue["state"] == "OPEN":
                actions.append("close")
            if not closed and issue["state"] == "CLOSED":
                actions.append("reopen")
        where = "new issue" if issue is None else f"#{issue['number']}"
        print(f"{m['id']:3} {where:>9}: {', '.join(actions) or 'up to date'}")
        drift |= bool(actions)
        if not args.apply:
            continue
        if issue is None:
            url = gh("issue", "create", "--title", title(m), "--body-file", "-",
                     *[a for l in LABELS for a in ("--label", l)], stdin=text).strip()
            number = url.rsplit("/", 1)[-1]
            print(f"    created {url}")
        else:
            number = str(issue["number"])
            if "edit" in actions:
                gh("issue", "edit", number, "--title", title(m), "--body-file", "-",
                   stdin=text)
        if "close" in actions:
            gh("issue", "close", number, "--reason", "completed")
        if "reopen" in actions:
            gh("issue", "reopen", number)
    if args.check and drift:
        sys.exit(1)


if __name__ == "__main__":
    main()
