#!/bin/zsh
# Copy the latest app from Google Drive into this site, commit it, and push (if possible).
set -e
cd "$(dirname "$0")"
SRC="/Users/wanchana/Library/CloudStorage/GoogleDrive-wanchana.pon@ceb-rama.org/My Drive/claude/gold-hourly/index_v2.html"
python3 - "$SRC" index.html <<'PY'
import sys
s = open(sys.argv[1], encoding='utf-8').read()
if 'name="robots"' not in s:
    s = s.replace('<meta name="viewport"', '<meta name="robots" content="noindex">\n<meta name="viewport"', 1)
open(sys.argv[2], 'w', encoding='utf-8').write(s)
PY
git add -A
if git diff --cached --quiet; then echo "No changes to publish."; else git commit -q -m "Update app $(date '+%Y-%m-%d %H:%M')"; echo "Committed."; fi
if git remote get-url origin >/dev/null 2>&1; then
  git push -q origin main && echo "Pushed: the site updates in about a minute." || echo "Push needs GitHub Desktop: open it and click 'Push origin'."
else
  echo "No GitHub remote yet: publish the repository from GitHub Desktop first."
fi
