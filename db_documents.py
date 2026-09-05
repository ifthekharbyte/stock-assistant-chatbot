"""
db_documents.py

Persistence for the knowledge base -- follows the same shape as
db_save.py/db_feedback.py, but for the `documents` table instead of
conversations/feedback.

ingest_kb.py writes here. knowledge_base.py reads everything back out
at startup and builds the in-memory keyword + vector indexes from it
(same pattern as the course: Postgres/SQLite is the durable store,
minsearch/numpy is the runtime index -- see 02-vector-search/lessons/
07-sqlitesearch-vector.md for the same split).
"""

from datetime import datetime

from db_init import DB_TIMEZONE, get_db_connection


def upsert_document(doc: dict) -> None:
    """
    doc needs: id, ticker, doc_type, title, text, source_url (optional),
    published_at (optional datetime).

    Re-running ingestion for the same id updates the row in place instead
    of creating duplicates, so ingest_kb.py is safe to re-run on a
    schedule.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (
                    id, ticker, doc_type, title, text, source_url,
                    published_at, ingested_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (id) DO UPDATE SET
                    ticker = EXCLUDED.ticker,
                    doc_type = EXCLUDED.doc_type,
                    title = EXCLUDED.title,
                    text = EXCLUDED.text,
                    source_url = EXCLUDED.source_url,
                    published_at = EXCLUDED.published_at,
                    ingested_at = EXCLUDED.ingested_at
                """,
                (
                    doc["id"],
                    doc["ticker"],
                    doc["doc_type"],
                    doc["title"],
                    doc["text"],
                    doc.get("source_url"),
                    doc.get("published_at"),
                    datetime.now(DB_TIMEZONE),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def upsert_documents(docs: list[dict]) -> None:
    for doc in docs:
        upsert_document(doc)


def get_all_documents() -> list[dict]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, ticker, doc_type, title, text, source_url, published_at FROM documents"
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {
            "id": row[0],
            "ticker": row[1],
            "doc_type": row[2],
            "title": row[3],
            "text": row[4],
            "source_url": row[5],
            "published_at": row[6],
        }
        for row in rows
    ]


def count_documents() -> int:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents")
            return cur.fetchone()[0]
    finally:
        conn.close()


if __name__ == "__main__":
    print(f"{count_documents()} documents in the knowledge base")
