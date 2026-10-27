#!/bin/sh
# Rebuild both editions and the service worker. Run from the repo root.
set -e
cd "$(dirname "$0")/.."
python3 tools/build_page.py tools/templates/template-ar.html book-ar.md index.html
python3 tools/build_page.py tools/templates/template-en.html book.md english.html
python3 tools/gen_sw.py
