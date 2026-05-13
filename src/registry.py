"""SQLite registry construction and summary helpers for local skill indexes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.embeddings import TextEmbedder
from core.sections import parse_markdown_sections, token_count
from core.views import build_skill_views
from loader import SkillLoadError, load_skill_document
from schema import SkillSpec


SCHEMA_VERSION = 2


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

        CREATE TABLE IF NOT EXISTS skill_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            skill_id TEXT NOT NULL,
            view_name TEXT NOT NULL,
            model_name TEXT NOT NULL,
            vector_index_name TEXT NOT NULL,
            vector_position INTEGER NOT NULL,
            source_text TEXT NOT NULL,
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
            DELETE FROM skill_embeddings;
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


def persist_dense_embeddings(
    skills: list[SkillSpec],
    db_path: str | Path,
    index_dir: str | Path,
    embedder: TextEmbedder,
    *,
    vector_index_name: str = "dense_views",
) -> dict[str, int | str]:
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    vector_path = index_path / f"{vector_index_name}.jsonl"
    id_map_path = index_path / f"{vector_index_name}_id_map.json"

    view_rows = [view for skill in skills for view in build_skill_views(skill)]
    vectors = embedder.embed_texts([view.text for view in view_rows])
    id_map = []

    with vector_path.open("w", encoding="utf-8") as vector_file, connect(db_path) as conn:
        initialize_registry(conn)
        conn.execute("DELETE FROM skill_embeddings WHERE vector_index_name = ?", (vector_index_name,))
        for position, (view, vector) in enumerate(zip(view_rows, vectors)):
            vector_file.write(
                json.dumps(
                    {
                        "position": position,
                        "skill_id": view.skill_id,
                        "view_name": view.view_name,
                        "vector": vector,
                    }
                )
                + "\n"
            )
            id_map.append({"position": position, "skill_id": view.skill_id, "view_name": view.view_name})
            conn.execute(
                """
                INSERT INTO skill_embeddings (
                    skill_id, view_name, model_name, vector_index_name,
                    vector_position, source_text
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (view.skill_id, view.view_name, embedder.model_name, vector_index_name, position, view.text),
            )
        conn.commit()

    id_map_path.write_text(json.dumps(id_map, indent=2), encoding="utf-8")
    return {
        "embedding_count": len(view_rows),
        "embedding_model": embedder.model_name,
        "vector_path": str(vector_path),
        "id_map_path": str(id_map_path),
    }


def registry_summary(db_path: str | Path) -> dict[str, int | str]:
    with connect(db_path) as conn:
        initialize_registry(conn)
        counts = {}
        for table in ("skills", "skill_documents", "skill_sections", "skill_views", "skill_embeddings"):
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        version = conn.execute("SELECT value FROM registry_meta WHERE key = 'schema_version'").fetchone()
    return {"schema_version": version[0] if version else "", **counts}


def dump_json_summary(summary: dict[str, int | str]) -> str:
    return json.dumps(summary, indent=2)
