#!/bin/zsh
# Publish the local index.html (this folder is the working copy), keep a backup copy in Google Drive, commit and push.
set -e
cd "$(dirname "$0")"
BACKUP="/Users/wanchana/Library/CloudStorage/GoogleDrive-wanchana.pon@ceb-rama.org/My Drive/claude/gold-hourly/index_v2.html"
cp index.html "$BACKUP" && echo "Backup copied to Google Drive (index_v2.html)."
git add -A
if git diff --cached --quiet; then echo "No changes to publish."; else git commit -q -m "Update app $(date '+%Y-%m-%d %H:%M')"; echo "Committed."; fi
git push -q origin main && echo "Pushed: the site updates in about a minute." || echo "Push needs GitHub Desktop: open it and click 'Push origin'."
