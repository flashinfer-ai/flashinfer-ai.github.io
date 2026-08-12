# flashinfer.ai

Source for the [FlashInfer project website](https://flashinfer.ai), built with
Jekyll and served by GitHub Pages from the `main` branch.

## Local preview

```bash
bundle install
bundle exec jekyll serve   # http://localhost:4000
```

Production uses GitHub Pages' default Jekyll build, not this repo's Gemfile, so
avoid Jekyll-4-only features. For a Pages-compatible build — the same image and
Jekyll version the Pages builder uses, though the `latest` tag moves and Pages
runs its own deployment:

```bash
docker run --rm -v "$PWD":/src -w /src --entrypoint jekyll \
  ghcr.io/actions/jekyll-build-pages:latest build -s . -d _site
```

## Content

### Blog posts

Markdown files in `_posts/`, named `YYYY-MM-DD-slug.md`. They appear on the
home page and in the RSS feed.

### Release highlights

The [Releases page](https://flashinfer.ai/releases/) collects the hand-written
highlights we prepend to the auto-generated changelog on each
[GitHub release](https://github.com/flashinfer-ai/flashinfer/releases). One file
per release in `_releases/`, all rendered into a single continuous page with a
sticky version nav by `_layouts/releases.html`.

Publish a release's highlights after tagging:

```bash
./scripts/import_release.py v0.6.17   # writes _releases/v0.6.17.md
```

The script pulls the release body with `gh`, drops the auto-generated changelog,
promotes the `**bold**` subsection headings to `###`, and turns bare PR
references into one linked row each, with the PR title alongside. Review the
result — prose written for a GitHub release sometimes needs light editing to
stand alone — then commit.

Entries sort by the `date` in their front matter, newest first, so only that
field and `tag` are required:

```markdown
---
tag: v0.6.17
date: 2026-08-11 05:09:14 +0000
---

One-paragraph summary of the release; rendered as the lead.

### Kimi K3 MLA decode on Blackwell

Section titles state a capability. Lead with what users can now run, then add
only enough mechanism to make the claim credible.

<ul class="pr-list">
<li><a href="https://github.com/flashinfer-ai/flashinfer/pull/4178">#4178</a> <span class="pr-title">feat(mla): support packed low-head and variable-Q decode</span></li>
</ul>
```

Releases whose GitHub notes are only an auto-generated changelog are left off
the page — it exists for the editorial summary, not to mirror every tag.
