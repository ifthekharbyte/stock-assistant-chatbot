"""
monitoring/dashboard.py

Streamlit dashboard over the Postgres tables written by db_save.py /
db_feedback.py. Starts from 05-monitoring/code/dashboard.py's structure
(headline metrics + line charts + recent conversations), and adds the
panels described in 05-monitoring/lessons/12-grafana.md (tool usage,
judge relevance breakdown, user thumbs up/down) as Streamlit charts
instead of Grafana panels, since -- per that same lesson -- "if your
needs are simple, the Streamlit dashboard is enough." Point docker-compose's
`grafana` service at the same Postgres database and build those exact
panels there if you want alerting or more advanced visualization.

Run with: streamlit run monitoring/dashboard.py
"""

import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from db_query import (  # noqa: E402
    get_conversations,
    get_relevance_breakdown,
    get_stats,
    get_tool_usage,
    get_user_feedback_breakdown,
)

st.set_page_config(page_title="Stock Assistant Dashboard", page_icon="📊", layout="wide")
st.title("📊 Stock Assistant — Monitoring")

stats = get_stats()

if stats.total == 0:
    st.info("No conversations logged yet. Chat with the assistant in app.py first.")
    st.stop()

# ---- Row 1: headline numbers (matches the course's dashboard.py) -----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total conversations", stats.total)
c2.metric("Avg response time", f"{stats.avg_response_time:.2f}s")
c3.metric("Total cost", f"${stats.total_cost:.4f}")
c4.metric("Avg tokens", f"{stats.avg_tokens:.0f}")

st.markdown("---")

records = [record for _, record in get_conversations(limit=200)]
df = pd.DataFrame([asdict(r) for r in records])

# ---- Chart 1 & 2: cost and response time over time --------------------
st.subheader("Cost over time")
st.line_chart(df, x="timestamp", y="cost")

st.subheader("Response time over time")
st.line_chart(df, x="timestamp", y="response_time")

col_a, col_b = st.columns(2)

# ---- Chart 3: tool usage frequency (see 12-grafana.md "Model Usage Panel") --
with col_a:
    st.subheader("Tool usage frequency")
    tool_usage = get_tool_usage()
    if tool_usage:
        tool_df = pd.DataFrame(sorted(tool_usage.items()), columns=["tool", "calls"]).set_index("tool")
        st.bar_chart(tool_df)
    else:
        st.caption("No tool calls logged yet.")

# ---- Chart 4: judge relevance breakdown ("Relevance Distribution Panel") ----
with col_b:
    st.subheader("Judge relevance breakdown")
    relevance = get_relevance_breakdown()
    if relevance:
        rel_df = pd.DataFrame(sorted(relevance.items()), columns=["relevance", "count"]).set_index("relevance")
        st.bar_chart(rel_df)
    else:
        st.caption("No judge feedback logged yet.")

# ---- Chart 5: user feedback ("User Feedback Panel") --------------------
st.subheader("User feedback (👍 / 👎)")
user_feedback = get_user_feedback_breakdown()
feedback_df = pd.DataFrame(
    {"feedback": ["👍 Positive", "👎 Negative"], "count": [user_feedback["thumbs_up"], user_feedback["thumbs_down"]]}
).set_index("feedback")
st.bar_chart(feedback_df)

st.markdown("---")
st.subheader("Recent conversations")
for record in records[:20]:
    st.write(f"**{record.question[:80]}**")
    st.write(f"{record.answer[:300]}{'...' if len(record.answer) > 300 else ''}")
    st.caption(f"⏱ {record.response_time:.2f}s · 💵 ${record.cost:.4f} · 🔧 {[c['name'] for c in record.tool_calls]}")
    st.divider()
