# Working on this repo

Build constraints and structure. See `README.md` for what the site contains.

## The build is not the one in the Gemfile

There is no Actions workflow, so GitHub Pages runs its **default** build and
ignores `Gemfile`. Production is Jekyll 3.x; the Gemfile's Jekyll 4.3 is only
what a local `bundle exec jekyll serve` would use. **Avoid Jekyll-4-only
features** — they work locally and break on deploy.

To build what Pages builds:

```bash
docker run --rm -v "$PWD":/src -w /src --entrypoint jekyll \
  ghcr.io/actions/jekyll-build-pages:latest build -s . -d _site
```

Two gotchas with that image: it aborts if a `Gemfile` is visible (build from a
copy without it, or set `JEKYLL_NO_BUNDLER_REQUIRE=true`), and system Ruby on
macOS is usually too old for the repo's own bundle, so Docker is often the only
option locally.

## Structure

| Path | Role |
|---|---|
| `pages/releases.md` | the page at `/releases/` |
| `_releases/*.md` | one entry per release (see `_releases/_README.md`) |
| `_layouts/releases.html` | renders the entries and the version nav |
| `_sass/releases.scss` | styles, imported from `_sass/minima.scss` |
| `scripts/import_release.py` | writes an entry from a GitHub release |

Things that are load-bearing and easy to undo by accident:

- **`_releases` is a collection with `output: false`.** Entries produce no pages
  on their own; `pages/releases.md` renders them. Files there whose names start
  with `_` are skipped entirely, which is why `_releases/_README.md` is not an
  entry.
- **`permalink: /releases/` sets the URL.** `pages/` means nothing to Jekyll, so
  without that line the page moves to `/pages/releases.html`.
- **The header nav is generated.** `_includes/header.html` iterates `site.pages`
  and renders any page with a title, using `nav_title` when set. A page moved
  into a collection or a `_`-prefixed directory disappears from the nav.
- **`body_class` in front matter** puts the wider layout and typography on the
  releases page only, so the blog and home page keep the default width.
- **`_sass/releases.scss` is additive.** It adds no rules to existing selectors;
  keep it that way so other pages cannot regress.

## Before pushing

Build with the Docker command above and diff `_site` against a build of `main`.
Existing pages should be unchanged apart from the feed's build timestamp.
