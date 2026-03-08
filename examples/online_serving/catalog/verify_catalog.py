#!/usr/bin/env python3
import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Verify generated results against a catalog.")
    parser.add_argument("result_path", help="Path to the result JSON file.")
    parser.add_argument("catalog_path", help="Path to the catalog JSON file.")
    args = parser.parse_args()

    RESULT_PATH = args.result_path
    CATALOG_PATH = args.catalog_path

    # Check if files exist
    if not os.path.exists(RESULT_PATH):
        print(f"Error: Result file not found at {RESULT_PATH}")
        sys.exit(1)
    if not os.path.exists(CATALOG_PATH):
        print(f"Error: Catalog file not found at {CATALOG_PATH}")
        sys.exit(1)

    # Load results
    print(f"Loading results from {RESULT_PATH}...")
    try:
        with open(RESULT_PATH, "r") as f:
            content = f.read()
        start_idx = content.find("{")
        if start_idx == -1:
            print("Error: No JSON object found in result file")
            sys.exit(1)
        decoder = json.JSONDecoder()
        results, _ = decoder.raw_decode(content[start_idx:])
    except json.JSONDecodeError as e:
        print(f"Error decoding result JSON: {e}")
        sys.exit(1)

    # Load catalog
    print(f"Loading catalog from {CATALOG_PATH}...")
    try:
        with open(CATALOG_PATH, "r") as f:
            catalog = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding catalog JSON: {e}")
        sys.exit(1)

    # Prepare catalog set for O(1) lookup
    # Assuming catalog is a list of lists of strings
    # We join inner lists to strings to match content format
    try:
        catalog_set = set("".join(item) for item in catalog)
    except TypeError:
        print("Error: Catalog format not supported. Expected a list of lists (or iterables).")
        sys.exit(1)

    print(f"Catalog loaded with {len(catalog_set)} items.")
    print(f"First 10 items in catalog: {list(catalog_set)[:10]}")

    # Verify choices
    choices = results.get("choices", [])
    print(f"Verifying {len(choices)} generated choices...")

    failures = []

    for i, choice in enumerate(choices):
        content = choice.get("message", {}).get("content", "")
        if not content:
            print(f"Warning: Choice {i} has no content.")
            continue

        if content not in catalog_set:
            failures.append((i, content, content))

    # Report results
    if failures:
        print(f"\nFAILURE: {len(failures)} items generated were NOT found in the catalog.")
        for idx, content, item in failures[:10]:
            print(f"  Choice {idx}: {item} (from '{content}')")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more.")
        sys.exit(1)
    else:
        print("\nSUCCESS: All generated items exist in the catalog.")


if __name__ == "__main__":
    main()
