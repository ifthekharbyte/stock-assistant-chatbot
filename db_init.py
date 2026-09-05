"""
db_init.py

Database connection and schema setup -- ported from
05-monitoring/code/db_init.py, using Postgres via psycopg (v3) instead
of SQLite so this matches the course's monitoring stack (Postgres +
Grafana + Streamlit dashboard, wired up in docker-compose.yaml).

Schema differences from the course's FAQ assistant, all additive:
- conversations gains a `tool_calls` column (JSON text) -- the agent's
  answers are backed by Massive API tool calls, not a fixed FAQ lookup,
  so we log which tools were called with which arguments.
- feedback keeps the exact course design: one table for both the
  automatic judge verdict (source='judge') and user thumbs up/down
  (source='user'), distinguished by `source`.
- a new `documents` table, absent from the course's FAQ assistant
  entirely: it's the persistent knowledge base that ingest_kb.py
  populates and knowledge_base.py loads into memory for keyword/vector/
  hybrid search (see 02-vector-search and 06-best-practices). This is
  what turns "connect to a live API" into "knowledge base + LLM" for
  the project rubric's retrieval-flow criterion.
"""

import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

DB_TIMEZONE = datetime.now().astimezone().tzinfo


def get_db_connection():
    import psycopg

    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        dbname=os.getenv("POSTGRES_DB", "stock_assistant"),
        user=os.getenv("POSTGRES_USER", "user"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
    )


def init_db(drop: bool = False):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if drop:
                cur.execute("DROP TABLE IF EXISTS feedback")
                cur.execute("DROP TABLE IF EXISTS conversations")
                cur.execute("DROP TABLE IF EXISTS documents")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    model TEXT NOT NULL,
                    instructions TEXT NOT NULL,
                    tool_calls TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    response_time FLOAT NOT NULL,
                    cost FLOAT NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id INTEGER REFERENCES conversations(id),
                    source TEXT NOT NULL,
                    relevance TEXT,
                    explanation TEXT,
                    score INTEGER,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    doc_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source_url TEXT,
                    published_at TIMESTAMP WITH TIME ZONE,
                    ingested_at TIMESTAMP WITH TIME ZONE NOT NULL
                )
                """
            )
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    init_db(drop=True)
    print("Database initialized")
