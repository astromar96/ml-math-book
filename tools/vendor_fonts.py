import re, sys, pathlib, urllib.request

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
KEEP = {"arabic", "latin", "latin-ext"}

out_dir = pathlib.Path(sys.argv[1])
font_dir = out_dir / "fonts"
font_dir.mkdir(parents=True, exist_ok=True)

def get(url, binary=False):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read() if binary else r.read().decode("utf-8")

# css name -> google fonts url
SHEETS = {
    "fonts-ar.css": ("https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;500;600;700"
                     "&family=Cairo:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap"),
    "fonts-en.css": ("https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;"
                     "0,8..60,600;0,8..60,700;1,8..60,400&family=Inter:wght@400;500;600"
                     "&family=JetBrains+Mono:wght@400;500&family=Noto+Naskh+Arabic:wght@400;600&display=swap"),
}

block_re = re.compile(r"/\*\s*([a-z0-9\-\[\]]+)\s*\*/\s*(@font-face\s*\{[^}]*\})", re.I)
url_re = re.compile(r"url\((https://fonts\.gstatic\.com/[^)]+)\)")
downloaded = {}

for css_name, css_url in SHEETS.items():
    css = get(css_url)
    kept, dropped = [], []
    for subset, block in block_re.findall(css):
        if subset.lower() not in KEEP:
            dropped.append(subset)
            continue
        def repl(m):
            u = m.group(1)
            if u not in downloaded:
                name = u.rsplit("/", 1)[-1]
                if not name.endswith(".woff2"):
                    name += ".woff2"
                # gstatic basenames can repeat across families; disambiguate
                stem, ext = name[:-6], ".woff2"
                n, final = 0, stem + ext
                while final in downloaded.values():
                    n += 1
                    final = "%s-%d%s" % (stem, n, ext)
                (font_dir / final).write_bytes(get(u, binary=True))
                downloaded[u] = final
            return "url(fonts/%s)" % downloaded[u]
        kept.append("/* %s */\n%s" % (subset, url_re.sub(repl, block)))
    header = ("/* Vendored from Google Fonts for offline use. Subsets kept: %s.\n"
              "   Source: %s */\n" % (", ".join(sorted(KEEP)), css_url))
    (out_dir / css_name).write_text(header + "\n".join(kept) + "\n", encoding="utf-8")
    print("%s: %d faces kept, %d subsets dropped" % (css_name, len(kept), len(dropped)))

print("woff2 files: %d" % len(set(downloaded.values())))
