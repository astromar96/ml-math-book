import pathlib, sys
tpl_path, md_path, out_path = (pathlib.Path(p) for p in sys.argv[1:4])
tpl = tpl_path.read_text(encoding="utf-8")
md = md_path.read_text(encoding="utf-8")
assert "</script" not in md.lower(), "markdown would close the embedding script tag"
assert tpl.count("__BOOK_MARKDOWN__") == 1
out_path.write_text(tpl.replace("__BOOK_MARKDOWN__", md), encoding="utf-8")
print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")
