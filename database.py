
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "tickets.db"


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_text TEXT NOT NULL,
                predicted_category TEXT NOT NULL,
                predicted_priority TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def save_ticket(ticket_text, category, priority):
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO tickets (
                ticket_text,
                predicted_category,
                predicted_priority
            )
            VALUES (?, ?, ?)
            """,
            (ticket_text, category, priority),
        )
        return cursor.lastrowid


def get_all_tickets():
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM tickets ORDER BY id DESC"
        ).fetchall()
        return [dict(row) for row in rows]


if __name__ == "__main__":
    initialize_database()
    print(f"Database initialized: {DB_PATH}")
