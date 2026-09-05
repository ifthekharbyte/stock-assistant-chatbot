"""
db_save.py

Ported from 05-monitoring/code/db_save.py. Takes the AgentCallRecord
produced by agent.agent_loop() (see agent.py) and writes one row to
`conversations`.
"""

import json
from datetime import datetime

from db_init import DB_TIMEZONE, get_db_connection


def save_conversation(record, question: str) -> int:
    timestamp = datetime.now(DB_TIMEZONE)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations (
                    question, answer, model, instructions, tool_calls,
                    prompt_tokens, completion_tokens, total_tokens,
                    response_time, cost, timestamp
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                RETURNING id
                """,
                (
                    question,
                    record.answer,
                    record.model,
                    record.instructions,
                    json.dumps(record.tool_calls),
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.total_tokens,
                    record.response_time,
                    record.cost,
                    timestamp,
                ),
            )
            conversation_id = cur.fetchone()[0]
        conn.commit()
    finally:
        conn.close()
    return conversation_id
