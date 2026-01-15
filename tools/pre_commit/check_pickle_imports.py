#!/usr/bin/env python3
"""
Pre-commit hook to prevent new pickle/cloudpickle imports.

This script checks for pickle or cloudpickle imports in staged Python files
and fails if any are found, as pickle can execute arbitrary code and is
a security risk.
"""

import re
import sys
from pathlib import Path

# Patterns to match pickle/cloudpickle imports
PICKLE_PATTERNS = [
    r"^\s*import\s+pickle",
    r"^\s*from\s+pickle\s+import",
    r"^\s*import\s+cloudpickle",
    r"^\s*from\s+cloudpickle\s+import",
]

# Compile regex patterns
compiled_patterns = [re.compile(pattern) for pattern in PICKLE_PATTERNS]


def check_file(file_path: Path) -> list[tuple[int, str]]:
    """
    Check a file for pickle/cloudpickle imports.

    Returns:
        List of tuples (line_number, line_content) for violations
    """
    violations = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                for pattern in compiled_patterns:
                    if pattern.match(line):
                        violations.append((line_num, line.rstrip()))
                        break
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return violations

    return violations


def main():
    """Main entry point for the pre-commit hook."""
    # Get files from stdin (pre-commit passes filenames via stdin)
    if len(sys.argv) > 1:
        files = [Path(f) for f in sys.argv[1:]]
    else:
        # If no arguments, read from stdin (pre-commit default behavior)
        files = [Path(line.strip()) for line in sys.stdin if line.strip()]

    all_violations = []

    for file_path in files:
        # Only check Python files
        if not file_path.suffix == ".py":
            continue

        if not file_path.exists():
            continue

        violations = check_file(file_path)
        if violations:
            all_violations.append((file_path, violations))

    if all_violations:
        print("ERROR: Found pickle/cloudpickle imports:", file=sys.stderr)
        print("", file=sys.stderr)
        print(
            "Pickle and cloudpickle can execute arbitrary code and are security risks.",
            file=sys.stderr,
        )
        print("Please use safer alternatives like json, msgpack, or protobuf.", file=sys.stderr)
        print("", file=sys.stderr)

        for file_path, violations in all_violations:
            print(f"{file_path}:", file=sys.stderr)
            for line_num, line_content in violations:
                print(f"  Line {line_num}: {line_content}", file=sys.stderr)

        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
