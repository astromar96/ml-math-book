# Build tooling

`index.html` and `english.html` are generated. Edit the Markdown or the
templates, never the generated pages.

```sh
sh tools/build.sh
```

That runs three steps:

| Step | What it does |
| --- | --- |
| `build_page.py` | drops `book-ar.md` into `templates/template-ar.html` → `index.html`, and `book.md` into `templates/template-en.html` → `english.html` |
| `gen_sw.py` | rewrites `sw.js`: rebuilds the precache list from what is actually on disk, and keys the cache to a hash of those files |

Because the cache name is a content hash, any change to the book, the templates
or `vendor/` ships as a new cache automatically — there is no version to bump by
hand. `gen_sw.py` refuses to write if a precache entry is missing from disk.

## Re-vendoring the fonts

`vendor/fonts/` and `vendor/fonts-*.css` are Google Fonts, downloaded so the
book needs no network. The CSS is Google's own with only the `url()` paths
rewritten, keeping the latin, latin-ext and arabic subsets.

```sh
python3 tools/vendor_fonts.py vendor
sh tools/build.sh
```

KaTeX and marked in `vendor/` come straight from their npm tarballs
(`katex@0.16.9`, `marked@12.0.2`); `vendor/katex/fonts/` must stay next to
`katex.min.css`, which references it relatively.

## Icons

`icons/icon.svg` is the master. The PNGs are rendered from it at 512px and
downscaled — Chrome clamps `--window-size` below ~500px, so rendering the small
sizes directly produces a blank image.

```sh
# render 512 from icons/icon.svg, then
python3 -c "from PIL import Image; m=Image.open('icons/icon-512.png').convert('RGB'); \
[m.resize((s,s), Image.LANCZOS).save(p, optimize=True) for p,s in \
[('icons/icon-192.png',192),('icons/apple-touch-icon.png',180)]]"
```

## Checking the PWA locally

A service worker needs a secure context; `127.0.0.1` counts, `file://` does not.
Serve the repo, open it, then stop the server and reload — the book should come
up unchanged with the network gone.
