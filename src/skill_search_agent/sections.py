from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    title: str
    key: str
    content: str


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def normalize_section_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def parse_markdown_sections(text: str) -> list[Section]:
    sections: list[Section] = []
    current_title = "Overview"
    current_lines: list[str] = []

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    sections.append(Section(current_title, normalize_section_name(current_title), content))
            current_title = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append(Section(current_title, normalize_section_name(current_title), content))

    return sections


def token_count(text: str) -> int:
    return len(text.split())


def truncate_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text, False
    return " ".join(tokens[:max_tokens]), True
