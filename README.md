# flashinfer.ai

Source for the [FlashInfer project website](https://flashinfer.ai), served by
GitHub Pages from the `main` branch.

## Local preview

```bash
bundle install
bundle exec jekyll serve   # http://localhost:4000
```

See `CLAUDE.md` for build constraints and how the site is put together.

## Content

### Blog posts

Markdown or HTML files in `_posts/`, named `YYYY-MM-DD-slug.md` or
`YYYY-MM-DD-slug.html`. They appear on the home page, `/posts/`, and in the RSS
feed. The header order is configured by `header_pages` in `_config.yml`.

Set `body_class: technical-blog` and `toc: true` for the serif article layout
and automatic section navigation used by the v0.7 posts. Give HTML posts an
explicit `excerpt` when their body contains Liquid raw blocks.

For internal preview builds, `preview: true` disables the GitHub comment
widget. Override `url` with the preview origin so feed and metadata links
point to the preview.

### Release highlights

The [Releases page](https://flashinfer.ai/releases/) renders one entry per
release from `_releases/`. After a release is tagged:

```bash
./scripts/import_release.py <tag>   # writes _releases/<tag>.md
```

See `_releases/_README.md` for the entry format. What belongs in the highlights
is an editorial question, decided before they are published on the GitHub
release — not here, which is also why not every tag has an entry.
