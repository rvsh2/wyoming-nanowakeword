#!/usr/bin/env python3
"""Verify that every version declaration in the repo agrees.

The version lives in four places; a mismatch between the git tag and the
add-on config version has already broken add-on installs once (Supervisor
pulls <image>:<config.yaml version>).

Usage: check_versions.py [expected-tag]
    expected-tag: optional git tag (e.g. v0.1.0); its "v" prefix is stripped.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import tomllib

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    versions = {
        "pyproject.toml": tomllib.loads((REPO / "pyproject.toml").read_text())[
            "project"
        ]["version"],
        "wyoming_nanowakeword/__init__.py": re.search(
            r'__version__ = "([^"]+)"',
            (REPO / "wyoming_nanowakeword" / "__init__.py").read_text(),
        ).group(1),  # type: ignore[union-attr]
        "nanowakeword/config.yaml": re.search(
            r"^version:\s*(\S+)$",
            (REPO / "nanowakeword" / "config.yaml").read_text(),
            re.MULTILINE,
        ).group(1),  # type: ignore[union-attr]
        "custom_components/nanowakeword/manifest.json": json.loads(
            (REPO / "custom_components" / "nanowakeword" / "manifest.json").read_text()
        )["version"],
    }

    if len(sys.argv) > 1:
        versions["git tag"] = sys.argv[1].lstrip("v")

    if len(set(versions.values())) == 1:
        print(f"OK: version {next(iter(versions.values()))} everywhere")
        return 0

    print("Version mismatch:", file=sys.stderr)
    for source, version in versions.items():
        print(f"  {source}: {version}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
