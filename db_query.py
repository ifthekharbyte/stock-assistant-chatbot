"""
db_query.py

Ported from 05-monitoring/code/db_query.py (get_conversations, get_stats,
Stats dataclass, row_to_record), extended with three extra queries
(get_tool_usage, get_relevance_breakdown, get_user_feedback_breakdown)
so the Streamlit dashboard can show the tool-usage, judge-relevance, and
user-feedback panels described in the course's 12-grafana.md lesson,
even without running Grafana itself.
"""

import json
from collections import Counter
from dataclasses import dataclass

from db_init import get_db_connection
from models import AgentCallRecord


@dataclass
class Stats:
    total: int
    avg_response_time: float
    total_cost: float
    avg_tokens: float


def row_to_record(row) -> AgentCallRecord:
    return AgentCallRecord(
        model=row[3],
        question=row[1],
        instructions=row[4],
        answer=row[2],
        tool_calls=json.loads(row[5]),
        prompt_tokens=row[6],
        completion_tokens=row[7],
        total_tokens=row[8],
        response_time=row[9],
        cost=row[10],
        timestamp=row[11],
    )


def get_conversations(limit: int = 100):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, question, answer, model, instructions, tool_calls,
                       prompt_tokens, completion_tokens, total_tokens,
                       response_time, cost, timestamp
                FROM conversations
                ORDER BY timestamp DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return [(row[0], row_to_record(row)) for row in rows]


def get_stats() -> Stats:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*), AVG(response_time), SUM(cost), AVG(total_tokens)
                FROM conversations
                """
            )
            row = cur.fetchone()
    finally:
        conn.close()

    return Stats(
        total=row[0] or 0,
        avg_response_time=row[1] or 0.0,
        total_cost=row[2] or 0.0,
        avg_tokens=row[3] or 0.0,
    )


def get_tool_usage() -> Counter:
    """How often each tool has been called, across all logged conversations."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT tool_calls FROM conversations")
            rows = cur.fetchall()
    finally:
        conn.close()

    counter = Counter()
    for (raw,) in rows:
        for call in json.loads(raw or "[]"):
            counter[call["name"]] += 1
    return counter


def get_relevance_breakdown() -> Counter:
    """Judge relevance verdicts, i.e. feedback where source='judge'."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT relevance, COUNT(*) FROM feedback WHERE source = 'judge' GROUP BY relevance"
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return Counter({relevance: count for relevance, count in rows})


def get_user_feedback_breakdown() -> dict:
    """Thumbs up vs thumbs down, i.e. feedback where source='user'."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    SUM(CASE WHEN score > 0 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN score < 0 THEN 1 ELSE 0 END)
                FROM feedback
                WHERE source = 'user'
                """
            )
            up, down = cur.fetchone()
    finally:
        conn.close()
    return {"thumbs_up": up or 0, "thumbs_down": down or 0}


if __name__ == "__main__":
    for conv_id, record in get_conversations(limit=10):
        print(conv_id, record)
