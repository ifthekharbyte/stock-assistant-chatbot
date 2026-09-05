"""
models.py

Plain data types shared across the app, kept free of any side-effecting
imports (no LLM client, no Massive client). AgentCallRecord used to live in
agent.py, but db_query.py (used by the monitoring dashboard) only needs the
dataclass, not the agent loop -- importing it from agent.py dragged in
tools.py -> massive_client.py, which raises at import time if
MASSIVE_API_KEY isn't set. The dashboard has no reason to need that key.
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class AgentCallRecord:
    """
    Mirrors 05-monitoring/code/metrics.py's LLMCallRecord, with one
    addition (`tool_calls`) since this agent's "context" comes from
    live tool calls instead of a fixed search index.
    """

    model: str
    question: str
    instructions: str
    answer: str
    tool_calls: list
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time: float
    cost: float
    timestamp: datetime = field(default_factory=datetime.now)
