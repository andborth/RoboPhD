#!/usr/bin/env python3
"""
Problem analyzer - provides category-specific and code-pattern advice.
"""

import json
import re
import sys
from pathlib import Path


def extract_categories(reflection_path: Path) -> list[str]:
    """Extract category names from reflection.md."""
    if not reflection_path.exists():
        return []

    content = reflection_path.read_text()

    if "## Categories" not in content:
        return []

    section = content.split("## Categories")[1].split("##")[0]
    matches = re.findall(r'^\s*-?\s*\*\*([^*]+)\*\*:', section, re.MULTILINE)
    return matches


def normalize_category(name: str) -> str:
    """Normalize category name for matching.

    - Lowercase
    - Replace special characters with spaces (so "I/O" becomes "I O")
    - Collapse whitespace
    """
    name = name.lower()
    name = re.sub(r'[^a-z0-9\s]', ' ', name)  # Replace special chars with spaces
    name = ' '.join(name.split())  # Collapse whitespace
    return name


def build_category_index(advice_dir: Path) -> dict[str, Path]:
    """Build index mapping normalized names to advice file paths."""
    index = {}
    if not advice_dir.exists():
        return index

    for md_file in advice_dir.glob("*.md"):
        raw_name = md_file.stem
        normalized = normalize_category(raw_name)
        index[normalized] = md_file

    return index


def load_category_advice(category: str, category_index: dict[str, Path]) -> tuple[str, str] | None:
    """Load advice from category-specific .md file if it exists.

    Returns (matched_category_name, advice_content) or None.
    """
    normalized = normalize_category(category)
    if normalized in category_index:
        filepath = category_index[normalized]
        return (filepath.stem, filepath.read_text().strip())
    return None


def detect_code_patterns(solution_path: Path, patterns_file: Path) -> list[str]:
    """Detect code patterns in solution and return matching advice."""
    if not solution_path.exists() or not patterns_file.exists():
        return []

    solution = solution_path.read_text()

    with open(patterns_file) as f:
        config = json.load(f)

    advice_list = []
    for pattern in config.get("patterns", []):
        if re.search(pattern["regex"], solution):
            advice_list.append(pattern["advice"])

    return advice_list


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent
    reflection_path = Path("reflection.md")
    solution_path = Path("solution.py")
    output_dir = Path("tool_output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "analysis.txt"

    advice_dir = script_dir / "category_advice"
    patterns_file = script_dir / "code_patterns.json"

    # Build category index once
    category_index = build_category_index(advice_dir)

    lines = ["## Analysis\n"]

    # Category-based advice
    categories = extract_categories(reflection_path)
    category_advice_found = False
    for category in categories:
        result = load_category_advice(category, category_index)
        if result:
            matched_name, advice = result
            lines.append(f"### {matched_name}\n")  # Use canonical name from filename
            lines.append(advice)
            lines.append("")
            category_advice_found = True

    # Code pattern detection
    code_advice = detect_code_patterns(solution_path, patterns_file)
    if code_advice:
        lines.append("### Code Patterns Detected\n")
        for advice in code_advice:
            lines.append(f"- {advice}")
        lines.append("")

    # If nothing found, just output minimal header
    if not category_advice_found and not code_advice:
        lines = ["## Analysis\n", "No problem-specific advice triggered. Please focus on the general instructions below.\n"]

    output = "\n".join(lines)
    output_path.write_text(output)

    print(f"Analysis complete - wrote to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
