#!/usr/bin/env python3
# coding=utf-8

import argparse
import subprocess
import sys


def get_modified_files(directories):
    """Get a list of Python files that have been modified in the current git branch.

    Args:
        directories: List of directories to search in

    Returns:
        List of paths to modified Python files
    """
    try:
        # Get list of modified files using git diff
        diff_command = ["git", "diff", "--name-only", "--diff-filter=d", "HEAD"]
        diff_output = subprocess.check_output(diff_command, text=True).strip()

        # Get list of new untracked files using git ls-files
        new_command = ["git", "ls-files", "--others", "--exclude-standard"]
        new_output = subprocess.check_output(new_command, text=True).strip()

        # Combine results
        all_changed_files = []
        if diff_output:
            all_changed_files.extend(diff_output.split("\n"))
        if new_output:
            all_changed_files.extend(new_output.split("\n"))

        # Filter only .py files that belong to specified directories
        python_files = [
            f for f in all_changed_files if f.endswith(".py") and any(f.startswith(d) for d in directories)
        ]

        return python_files
    except subprocess.SubprocessError as e:
        print(f"Error executing git command: {e}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser(description="Get list of modified Python files")
    parser.add_argument("directories", nargs="+", help="List of directories to search in")
    args = parser.parse_args()

    modified_files = get_modified_files(args.directories)
    print(" ".join(modified_files))


if __name__ == "__main__":
    main()
