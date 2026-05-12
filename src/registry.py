"""SQLite registry construction and summary helpers for local skill indexes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.sections import parse_markdown_sections, token_count
from core.views import build_skill_views
from loader import SkillLoadError, load_skill_document
from schema import SkillSpec


SCHEMA_VERSION = 1


def default_db_path(index_dir: str | Path) -> Path:
    return Path(index_dir) / "skills.db"


def _sqlite3() -> Any:
    try:
        import sqlite3
    except ImportError as exc:
        raise SkillLoadError(f"SQLite is not available in this Python environment: {exc}") from exc

    return sqlite3


def connect(db_path: str | Path) -> Any:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sqlite3 = _sqlite3()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_registry(conn: Any) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS registry_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS skills (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            version TEXT NOT NULL,
            status TEXT NOT NULL,
            skill_type TEXT NOT NULL,
            interaction_mode TEXT NOT NULL,
            execution_available INTEGER NOT NULL,
            description TEXT NOT NULL,
            spec_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS skill_documents (
            skill_id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            content TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            FOREIGN KEY(skill_id) REFERENCES skills(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS skill_sections (
            skill_id TEXT NOT NULL,
            section_key TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            PRIMARY KEY(skill_id, section_key),
            FOREIGN KEY(skill_id) REFERENCES skills(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS skill_views (
            skill_id TEXT NOT NULL,
            view_name TEXT NOT NULL,
            text TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            PRIMARY KEY(skill_id, view_name),
            FOREIGN KEY(skill_id) REFERENCES skills(id) ON DELETE CASCADE
        );
        """
    )
    conn.execute(
        "INSERT OR REPLACE INTO registry_meta(key, value) VALUES (?, ?)",
        ("schema_version", str(SCHEMA_VERSION)),
    )


def rebuild_registry(skills: list[SkillSpec], db_path: str | Path) -> dict[str, int]:
    with connect(db_path) as conn:
        initialize_registry(conn)
        conn.executescript(
            """
            DELETE FROM skill_views;
            DELETE FROM skill_sections;
            DELETE FROM skill_documents;
            DELETE FROM skills;
            """
        )

        section_count = 0
        view_count = 0
        for skill in skills:
            conn.execute(
                """
                INSERT INTO skills (
                    id, name, version, status, skill_type, interaction_mode,
                    execution_available, description, spec_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    skill.id,
                    skill.name,
                    skill.version,
                    skill.status,
                    skill.skill_type.value,
                    skill.interaction.mode.value,
                    int(skill.execution_available),
                    skill.description.short,
                    skill.json(exclude={"root_dir"}),
                ),
            )

            document = load_skill_document(skill)
            conn.execute(
                """
                INSERT INTO skill_documents(skill_id, path, content, token_count)
                VALUES (?, ?, ?, ?)
                """,
                (skill.id, skill.content.path, document, token_count(document)),
            )

            for section in parse_markdown_sections(document):
                conn.execute(
                    """
                    INSERT INTO skill_sections(skill_id, section_key, title, content, token_count)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (skill.id, section.key, section.title, section.content, token_count(section.content)),
                )
                section_count += 1

            for view in build_skill_views(skill):
                conn.execute(
                    """
                    INSERT INTO skill_views(skill_id, view_name, text, token_count)
                    VALUES (?, ?, ?, ?)
                    """,
                    (view.skill_id, view.view_name, view.text, token_count(view.text)),
                )
                view_count += 1

        conn.commit()
        return {"skill_count": len(skills), "section_count": section_count, "view_count": view_count}


def registry_summary(db_path: str | Path) -> dict[str, int | str]:
    with connect(db_path) as conn:
        initialize_registry(conn)
        counts = {}
        for table in ("skills", "skill_documents", "skill_sections", "skill_views"):
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        version = conn.execute("SELECT value FROM registry_meta WHERE key = 'schema_version'").fetchone()
    return {"schema_version": version[0] if version else "", **counts}


def dump_json_summary(summary: dict[str, int | str]) -> str:
    return json.dumps(summary, indent=2)
