#!/usr/bin/env python3
# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check that source files have proper license headers."""

import sys
from pathlib import Path

LICENSE_HEADER = """Copyright (c) Nex-AGI. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""


def check_file(filepath: Path) -> bool:
    """Check if a file has the proper license header.

    Returns:
        True if the file has a valid license header, False otherwise.
    """
    try:
        content = filepath.read_text()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False

    # Skip empty files
    if not content.strip():
        return True

    # Remove comment markers and check for license header
    lines = content.split("\n")
    cleaned_lines = []

    for line in lines[:25]:  # Check first 25 lines
        # Remove comment markers
        stripped = line.strip()
        if stripped.startswith("#"):
            stripped = stripped[1:].strip()
        elif stripped.startswith("//"):
            stripped = stripped[2:].strip()

        cleaned_lines.append(stripped)

    cleaned_content = "\n".join(cleaned_lines)

    # Check for key phrases from the license
    if "Copyright (c) Nex-AGI" in cleaned_content and "Apache License" in cleaned_content:
        return True

    print(f"Missing or incorrect license header in: {filepath}")
    return False


def main():
    """Check license headers in all provided files."""
    if len(sys.argv) < 2:
        print("Usage: check_license_header.py <file1> <file2> ...")
        sys.exit(0)

    failed = []
    for filepath_str in sys.argv[1:]:
        filepath = Path(filepath_str)

        # Skip certain files
        if any(
            part in filepath.parts
            for part in [
                "__pycache__",
                ".pytest_cache",
                "wandb",
                ".git",
                "dist",
                "build",
                ".egg-info",
            ]
        ):
            continue

        # Skip generated files
        if filepath.name in ["__init__.py"] and filepath.stat().st_size < 100:
            continue

        if not check_file(filepath):
            failed.append(filepath)

    if failed:
        print(f"\n{len(failed)} file(s) missing proper license headers.")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
