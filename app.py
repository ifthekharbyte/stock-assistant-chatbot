"""
app.py

Chat interface for the stock market assistant. Follows
05-monitoring/code/app.py's pattern: every answer is (1) logged to
Postgres via db_save.save_conversation, (2) automatically scored by the
judge in judge.py and logged to `feedback` with source='judge', and
(3) open to a user thumbs up/down, logged to `feedback` with
source='user'. The course's app.py takes one question at a time;
this one keeps a multi-turn chat since the agent supports follow-ups
naturally (see agent.agent_loop's `previous_messages` parameter).

Run with: streamlit run app.py
"""

import streamlit as st

from agent import agent_loop
from db_feedback import save_feedback
from db_init import init_db
from db_save import save_conversation
from judge import evaluate_relevance
from tools import warm_up_knowledge_base

st.set_page_config(page_title="Stock Market Assistant", page_icon="📈")

init_db()
warm_up_knowledge_base()

st.title("📈 Stock Market Assistant")
st.caption("Live data via the Massive Stock Market API — ask about prices, trends, news, and more.")

with st.sidebar:
    st.markdown("### Try asking")
    st.markdown(
        "- What's the current price of AAPL?\n"
        "- Compare TSLA and RIVN over the last month\n"
        "- Is the market open right now?\n"
        "- What are today's top losers?\n"
        "- Any recent news on NVDA?\n"
        "- What's Coca-Cola's dividend history?"
    )
    st.markdown("---")
    st.caption("Not financial advice. Data may be delayed depending on your Massive plan.")

if "history" not in st.session_state:
    st.session_state.history = []  # list of {role, content} for display
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = None  # full Responses-API message history
if "last_conversation_id" not in st.session_state:
    st.session_state.last_conversation_id = None

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

question = st.chat_input("Ask about a stock, e.g. 'How is MSFT doing today?'")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Checking the markets..."):
            record, messages = agent_loop(question, previous_messages=st.session_state.agent_messages)
        st.markdown(record.answer)

        if record.tool_calls:
            with st.expander("Tools used"):
                for call in record.tool_calls:
                    st.code(f"{call['name']}({call['arguments']})", language="python")

        conversation_id = save_conversation(record, question)

        with st.spinner("Scoring relevance..."):
            relevance, explanation = evaluate_relevance(question, record.answer)
        save_feedback(conversation_id, "judge", relevance=relevance, explanation=explanation)

        badge = {"RELEVANT": "🟢", "PARTLY_RELEVANT": "🟡", "NON_RELEVANT": "🔴"}[relevance]
        st.caption(
            f"{badge} {relevance.replace('_', ' ').title()} · "
            f"{record.response_time:.1f}s · "
            f"{record.prompt_tokens}+{record.completion_tokens} tokens · "
            f"~${record.cost:.5f}"
        )

    st.session_state.history.append({"role": "assistant", "content": record.answer})
    st.session_state.agent_messages = messages
    st.session_state.last_conversation_id = conversation_id

if st.session_state.last_conversation_id:
    col1, col2, _ = st.columns([1, 1, 6])
    if col1.button("👍", key=f"up-{st.session_state.last_conversation_id}"):
        save_feedback(st.session_state.last_conversation_id, "user", score=1)
        st.toast("Thanks for the feedback!")
    if col2.button("👎", key=f"down-{st.session_state.last_conversation_id}"):
        save_feedback(st.session_state.last_conversation_id, "user", score=-1)
        st.toast("Thanks for the feedback!")
