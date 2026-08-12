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

The [Releases page](https://flashinfer.ai/releases/) renders one entry per
release from `_releases/`. After a release is tagged:

```bash
./scripts/import_release.py <tag>   # writes _releases/<tag>.md
```

See `_releases/_README.md` for the entry format. What belongs in the highlights
is an editorial question, decided before they are published on the GitHub
release — not here, which is also why not every tag has an entry.
