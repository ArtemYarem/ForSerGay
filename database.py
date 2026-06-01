"""
Database module — SQLite зберігання:
- Користувачі та їх API ключі
- Персонажі
- Історія розмов
"""

import sqlite3
import json
from typing import Optional

class Database:
    def __init__(self, path: str = "bot_data.db"):
        self.path = path
        self._init()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id   INTEGER PRIMARY KEY,
                    api_key   TEXT
                );

                CREATE TABLE IF NOT EXISTS characters (
                    user_id      INTEGER PRIMARY KEY,
                    name         TEXT NOT NULL,
                    appearance   TEXT NOT NULL,
                    personality  TEXT NOT NULL,
                    speech_style TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS history (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id  INTEGER NOT NULL,
                    role     TEXT NOT NULL,   -- 'user' або 'model'
                    content  TEXT NOT NULL,
                    ts       INTEGER DEFAULT (strftime('%s','now'))
                );

                CREATE INDEX IF NOT EXISTS idx_history_user
                    ON history(user_id, ts);
            """)

    # ── Користувачі ────────────────────────────────────────────────────────

    def ensure_user(self, user_id: int):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO users(user_id) VALUES(?)",
                (user_id,)
            )

    def set_api_key(self, user_id: int, key: str):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO users(user_id, api_key) VALUES(?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET api_key=excluded.api_key",
                (user_id, key)
            )

    def get_api_key(self, user_id: int) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT api_key FROM users WHERE user_id=?", (user_id,)
            ).fetchone()
            return row["api_key"] if row else None

    # ── Персонажі ──────────────────────────────────────────────────────────

    def save_character(self, user_id: int, data: dict):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO characters(user_id, name, appearance, personality, speech_style) "
                "VALUES(?,?,?,?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET "
                "  name=excluded.name, "
                "  appearance=excluded.appearance, "
                "  personality=excluded.personality, "
                "  speech_style=excluded.speech_style",
                (
                    user_id,
                    data["name"],
                    data["appearance"],
                    data["personality"],
                    data["speech_style"],
                )
            )

    def get_character(self, user_id: int) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT name, appearance, personality, speech_style "
                "FROM characters WHERE user_id=?",
                (user_id,)
            ).fetchone()
            return dict(row) if row else None

    # ── Історія ────────────────────────────────────────────────────────────

    def add_message(self, user_id: int, role: str, content: str):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO history(user_id, role, content) VALUES(?,?,?)",
                (user_id, role, content)
            )

    def get_history(self, user_id: int, limit: int = 30) -> list[tuple[str, str]]:
        """Повертає останні `limit` повідомлень у форматі [(role, content), ...]."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT role, content FROM ("
                "  SELECT role, content, ts FROM history "
                "  WHERE user_id=? ORDER BY ts DESC LIMIT ?"
                ") ORDER BY ts ASC",
                (user_id, limit)
            ).fetchall()
            return [(r["role"], r["content"]) for r in rows]

    def clear_history(self, user_id: int):
        with self._conn() as conn:
            conn.execute("DELETE FROM history WHERE user_id=?", (user_id,))
