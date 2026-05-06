#!/bin/bash
set -e

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Error: Working tree has uncommitted changes. Please commit or stash them first."
    exit 1
fi

# Configuration
UPSTREAM_REF="upstream/main"
EXPORT_BRANCH="export-branch-$(date +%Y%m%d-%H%M%S)"
PATCH_FILE=$(mktemp /tmp/changes.patch.XXXXXX)
ADDED_FILES_LIST=$(mktemp /tmp/added_files.txt.XXXXXX)
EXCLUDE_DIR="trc"

# Ensure we have the upstream ref
if ! git rev-parse --verify "$UPSTREAM_REF" >/dev/null 2>&1; then
    echo "Error: $UPSTREAM_REF not found. Please fetch upstream."
    exit 1
fi

echo "Starting export process..."

# 1. Create a diff between current version and upstream/main
# We use HEAD to represent the current version.
# We exclude the 'trc' folder using pathspec magic ':!trc'.
# We use --binary to ensure binary files are handled correctly.
echo '1. Creating diff (excluding' "$EXCLUDE_DIR" ')...'
git diff --binary "$UPSTREAM_REF" HEAD -- . ":!$EXCLUDE_DIR" > "$PATCH_FILE"

# 2. Create a list of added files
# We filter for Added (A) files.
echo "2. Listing added files..."
git diff --name-only --diff-filter=A "$UPSTREAM_REF" HEAD -- . ":!$EXCLUDE_DIR" > "$ADDED_FILES_LIST"

# 3. Create an export branch based on upstream/main
echo "3. Creating export branch $EXPORT_BRANCH..."
git checkout -b "$EXPORT_BRANCH" "$UPSTREAM_REF"

# 4. Apply the diff created in stage 1
echo "4. Applying diff..."
if [ -s "$PATCH_FILE" ]; then
    git apply "$PATCH_FILE"
else
    echo "No changes detected in patch file."
fi

# 5. Add the files created at stage 2
echo "5. Adding added files..."
if [ -s "$ADDED_FILES_LIST" ]; then
    while IFS= read -r file; do
        if [ -e "$file" ]; then
            git add "$file"
            echo "  Staged: $file"
        fi
    done < "$ADDED_FILES_LIST"
else
    echo "No added files to stage."
fi

# Cleanup
rm -f "$PATCH_FILE" "$ADDED_FILES_LIST"

echo "Done. Switched to $EXPORT_BRANCH."
