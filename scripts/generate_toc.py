#!/usr/bin/env python3
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

TOC_START = "<!-- toc:start -->"
TOC_END = "<!-- toc:end -->"

# Prefer explicit {#id} if present; otherwise make a GitHub-like slug
ID_ATTR_RE = re.compile(r"\s*\{#([A-Za-z0-9\-_]+)\}\s*$")
HEADING_RE = re.compile(r"^(#{2,6})\s+(.+)$")


def slugify(text: str) -> str:
    # Remove markdown inline code/backticks and links
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
    # Remove emojis and non-word symbols except spaces and hyphens
    text = re.sub(r"[:`~!@#$%^&*()=+\[\]{}|;:'\",.<>/?\\]", "", text)
    text = text.strip().lower()
    # Replace spaces with hyphens
    text = re.sub(r"\s+", "-", text)
    # Collapse multiple hyphens
    text = re.sub(r"-+", "-", text)
    return text


def parse_headings(lines):
    headings = []
    for i, line in enumerate(lines):
        m = HEADING_RE.match(line)
        if not m:
            continue
        level = len(m.group(1))
        title = m.group(2).strip()
        # ignore the TOC heading itself
        if "目錄" in title:
            continue
        # extract explicit id if exists
        id_attr = None
        m_id = ID_ATTR_RE.search(title)
        if m_id:
            id_attr = m_id.group(1)
            title = ID_ATTR_RE.sub("", title).strip()
        anchor = id_attr or slugify(title)
        headings.append((level, title, anchor))
    return headings


def build_toc(headings):
    toc = []
    for level, title, anchor in headings:
        # Only include level 2 and 3 to keep it compact
        if level == 2:
            toc.append(f"- [{title}](#{anchor})")
        elif level == 3:
            toc.append(f"  - [{title}](#{anchor})")
    return "\n".join(toc) + "\n"


def replace_toc(content: str, toc_block: str) -> str:
    if TOC_START in content and TOC_END in content:
        pre, rest = content.split(TOC_START, 1)
        _, post = rest.split(TOC_END, 1)
        new = pre + TOC_START + "\n" + "<!-- 目錄由 scripts/generate_toc.py 自動產生，請勿手動編輯 -->\n" + toc_block + TOC_END + post
        return new
    else:
        # If markers not found, insert after the first TOC heading
        lines = content.splitlines()
        for idx, line in enumerate(lines):
            if line.strip().startswith("## ") and "目錄" in line:
                insert_at = idx + 1
                lines.insert(insert_at, TOC_START)
                lines.insert(insert_at + 1, "<!-- 目錄由 scripts/generate_toc.py 自動產生，請勿手動編輯 -->")
                lines.insert(insert_at + 2, toc_block.rstrip())
                lines.insert(insert_at + 3, TOC_END)
                return "\n".join(lines) + "\n"
        return content


def main():
    content = README.read_text(encoding="utf-8")
    lines = content.splitlines()
    headings = parse_headings(lines)
    toc_block = build_toc(headings)
    new_content = replace_toc(content, toc_block)
    if new_content != content:
        README.write_text(new_content, encoding="utf-8")
        print("TOC updated in README.md")
    else:
        print("No changes to TOC")


if __name__ == "__main__":
    main()
