#!/usr/bin/env python3
"""Turn a GitHub release's hand-written highlights into a `_releases/` entry.

We write release highlights on the GitHub release page, prepended to the
auto-generated changelog. This script pulls that prose back out and normalizes
it for the website, so the two never drift.

    ./scripts/import_release.py v0.6.17          # writes _releases/v0.6.17.md
    ./scripts/import_release.py v0.6.17 --stdout # preview without writing

Normalization applied:
  * everything from "## What's Changed" onward (the auto-generated changelog,
    "New Contributors", "Full Changelog") is dropped
  * a leading "## Highlights" / "### v0.6.16 Highlights" heading is dropped --
    the page renders the version as the section heading already
  * "**Bold subsection**" paragraphs become "### Bold subsection", and any h1/h2
    is demoted to h3 -- the page already spends h1 on "Releases" and h2 on the
    version, so entries start at h3
  * bare PR references ("- #3852" or "- https://.../pull/3852"), which only
    auto-link on GitHub, become a row of real links styled as chips

Requires the `gh` CLI, authenticated.
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys

REPO = "flashinfer-ai/flashinfer"
ROOT = pathlib.Path(__file__).resolve().parent.parent

# "- #3852" or "- https://github.com/<owner>/<repo>/pull/3852"
PR_LINE = re.compile(
    r"^\s*[-*]\s*(?:#(?P<num>\d+)|https?://github\.com/[\w.-]+/[\w.-]+/pull/(?P<url_num>\d+))\s*$"
)
# A paragraph that is nothing but bold text, i.e. a subsection heading.
BOLD_HEADING = re.compile(r"^\*\*(?P<text>.+?)\*\*:?\s*$")
HIGHLIGHTS_HEADING = re.compile(r"^#+\s*(v?[\d.]+\s+)?highlights\s*$", re.IGNORECASE)
SHALLOW_HEADING = re.compile(r"^(#{1,2})\s+(?P<text>.+?)\s*$")
PR_PLACEHOLDER = "\x00PR_LIST:{}\x00"
CHANGELOG_HEADING = re.compile(r"^#+\s*(what'?s changed|new contributors)\s*$", re.IGNORECASE)


def fetch(tag):
    out = subprocess.run(
        ["gh", "release", "view", tag, "--repo", REPO, "--json", "body,publishedAt"],
        capture_output=True,
        text=True,
    )
    if out.returncode:
        sys.exit(f"gh failed for {tag}: {out.stderr.strip()}")
    data = json.loads(out.stdout)
    return data["body"].replace("\r\n", "\n"), data["publishedAt"]


def fetch_pr_titles(numbers):
    """Map PR number -> title, in as few round trips as possible."""
    owner, name = REPO.split("/")
    titles = {}
    numbers = sorted(set(numbers))
    # One aliased GraphQL query per chunk beats one REST call per PR.
    for start in range(0, len(numbers), 50):
        chunk = numbers[start : start + 50]
        fields = "\n".join(
            f'  p{n}: pullRequest(number: {n}) {{ title }}' for n in chunk
        )
        query = f'query {{ repository(owner: "{owner}", name: "{name}") {{\n{fields}\n}} }}'
        out = subprocess.run(
            ["gh", "api", "graphql", "-f", f"query={query}"],
            capture_output=True,
            text=True,
        )
        if out.returncode:
            sys.exit(f"gh graphql failed: {out.stderr.strip()}")
        repo_data = json.loads(out.stdout)["data"]["repository"]
        for n in chunk:
            node = repo_data.get(f"p{n}")
            if node:
                titles[n] = node["title"]
    return titles


def escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_pr_list(numbers, titles):
    """One PR per row: a pill holding the number, then the title beside it."""
    rows = []
    for n in numbers:
        # PR titles occasionally carry stray leading/trailing whitespace.
        title = (titles.get(n) or "").strip()
        label = f' <span class="pr-title">{escape(title)}</span>' if title else ""
        rows.append(
            f'<li><a href="https://github.com/{REPO}/pull/{n}">#{n}</a>{label}</li>'
        )
    return '<ul class="pr-list">\n' + "\n".join(rows) + "\n</ul>"


def convert(body):
    """Return (markdown, pr_runs).

    PR runs are left as placeholders so their titles can be fetched in one
    batch afterwards rather than one request per run.
    """
    lines = body.split("\n")
    out = []
    pr_runs = []
    i = 0
    in_code = False

    while i < len(lines):
        line = lines[i]

        if line.lstrip().startswith("```"):
            in_code = not in_code
            out.append(line.rstrip())
            i += 1
            continue

        if in_code:
            out.append(line.rstrip())
            i += 1
            continue

        if CHANGELOG_HEADING.match(line.strip()):
            break
        if line.strip().startswith("**Full Changelog**"):
            break
        if HIGHLIGHTS_HEADING.match(line.strip()):
            i += 1
            continue

        # Gather a run of PR references into one list.
        if PR_LINE.match(line):
            nums = []
            while i < len(lines):
                m = PR_LINE.match(lines[i])
                if not m:
                    break
                nums.append(int(m.group("num") or m.group("url_num")))
                i += 1
            pr_runs.append(nums)
            out.append(PR_PLACEHOLDER.format(len(pr_runs) - 1))
            continue

        heading = BOLD_HEADING.match(line.strip())
        if heading:
            out.append(f"### {heading.group('text')}")
            i += 1
            continue

        # h1/h2 would collide with the page's own "Releases" / version headings.
        shallow = SHALLOW_HEADING.match(line)
        if shallow:
            out.append(f"### {shallow.group('text')}")
            i += 1
            continue

        out.append(line.rstrip())
        i += 1

    # Collapse runs of blank lines and trim.
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, pr_runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag", help="release tag, e.g. v0.6.17")
    ap.add_argument("--stdout", action="store_true", help="print instead of writing the file")
    ap.add_argument("--force", action="store_true", help="overwrite an existing entry")
    args = ap.parse_args()

    body, published = fetch(args.tag)
    content, pr_runs = convert(body)
    if not content:
        sys.exit(
            f"{args.tag} has no hand-written highlights -- only the auto-generated "
            "changelog. Nothing to publish."
        )

    titles = fetch_pr_titles([n for run in pr_runs for n in run])
    for idx, run in enumerate(pr_runs):
        content = content.replace(PR_PLACEHOLDER.format(idx), render_pr_list(run, titles))

    # GitHub gives "2026-07-31T07:55:09Z"; Jekyll wants "2026-07-31 07:55:09 +0000".
    date = published.replace("T", " ").replace("Z", " +0000")
    doc = f"---\ntag: {args.tag}\ndate: {date}\n---\n\n{content}\n"

    if args.stdout:
        print(doc, end="")
        return

    dest = ROOT / "_releases" / f"{args.tag}.md"
    if dest.exists() and not args.force:
        sys.exit(f"{dest.relative_to(ROOT)} already exists; pass --force to overwrite.")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(doc)
    print(f"wrote {dest.relative_to(ROOT)}  ({len(content.splitlines())} lines)")
    print("Review it -- highlights often need light editing for a standalone page.")


if __name__ == "__main__":
    main()
