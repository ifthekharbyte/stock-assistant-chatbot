"""
db_feedback.py

Ported unchanged in spirit from 05-monitoring/code/db_feedback.py.
One function handles both kinds of feedback this app records:

- source='judge': automatic relevance verdict from judge.py
  (relevance, explanation set; score left null)
- source='user': a person's thumbs up/down in app.py
  (score set to +1/-1; relevance/explanation left null)
"""

from datetime import datetime

from db_init import DB_TIMEZONE, get_db_connection


def save_feedback(conversation_id: int, source: str, relevance: str | None = None,
                   explanation: str | None = None, score: int | None = None):
    timestamp = datetime.now(DB_TIMEZONE)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO feedback (
                    conversation_id, source, relevance, explanation, score, timestamp
                ) VALUES (
                    %s, %s, %s, %s, %s, %s
                )
                """,
                (conversation_id, source, relevance, explanation, score, timestamp),
            )
        conn.commit()
    finally:
        conn.close()
