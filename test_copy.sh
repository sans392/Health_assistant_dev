#!/usr/bin/env bash

set -e

N="$1"
DEST="$2"

mkdir -p "$DEST"

git log -n "$N" --name-only --pretty=format: --diff-filter=AM -m |
while read -r file; do
  [ -f "$file" ] && cp --parents "$file" "$DEST"
done

git log --name-status --oneline -"$N"

echo "Done: files copied to $DEST"
