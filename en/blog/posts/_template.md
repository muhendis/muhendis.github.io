<!--
  POST TEMPLATE — copy this file, rename it to your slug (lowercase, hyphens).

  Rules:
  1. Do NOT start with an "# H1" heading. The <h1> is rendered from the
     "title" field in posts.json so the heading, <title> and OG tags cannot
     drift apart. Start your content at "##".
  2. No YAML front matter. All metadata lives in posts.json.
  3. Add a matching entry to posts.json, where "slug" equals this filename
     without the .md extension. CI fails the build if they disagree.
  4. If you write the other-language version too, set "translationSlug" on
     BOTH entries to point at each other.
  5. Add the post URL to /sitemap.xml.
-->

Opening paragraph. This is what the reader sees first, so make it carry the point
rather than introduce it.

## First section

Body text. **Bold**, *italic*, `inline code` and [links](https://example.com) all work.

```python
# Fenced code blocks are supported and scroll horizontally if long.
import torch
```

- Bullet lists
- work as expected

## Second section

More content.
