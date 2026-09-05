"""
agent.py

The agentic core of the chatbot -- the stock-market equivalent of
01-agentic-rag/code/rag_helper.py + 05-monitoring/code/metrics.py
combined: it runs the function-calling agent loop from
01-agentic-rag/code/agents.ipynb, and returns an AgentCallRecord in the
same shape as the course's LLMCallRecord, so db_save.py/db_query.py can
log and read it the same way the course logs RAG calls.

Runs on Groq (see evaluation_utils.get_client()) via Groq's OpenAI-
compatible Responses API. That API mirrors OpenAI's closely enough that
the function-calling loop below is unchanged from an OpenAI-backed
version -- same tools= schema, same response.output item types
(function_call / message), same usage.input_tokens/output_tokens
fields. The one documented difference that matters here: Groq's
Responses API doesn't support stateful conversations via
previous_response_id, so the full message history must be resent every
turn -- which this loop already does (see `messages` below), so no
change was needed there either.

Two ways to run the agent, matching the notebook's progression:

1. `agent_loop()` -- framework-free, raw OpenAI-SDK-against-Groq with
   manual message-list bookkeeping (mirrors the notebook's final
   `agent_loop` cell). Used by app.py and eval/evaluate.py.

2. `build_toyaikit_runner()` -- the ToyAIKit-based version (mirrors the
   notebook's `OpenAIResponsesRunner` cells, and 04-evaluation's agent
   evaluation lesson). Handy for notebooks/explore_agent.ipynb.
"""

import json
import os
import re
import time

from dotenv import load_dotenv

from evaluation_utils import DEFAULT_MODEL, calc_price, get_client, responses_create_retry
from models import AgentCallRecord
from tools import TOOL_SCHEMAS, call_tool

load_dotenv()

MODEL = os.getenv("MODEL_NAME", DEFAULT_MODEL)

INSTRUCTIONS = """
You're a stock market research assistant backed by live data from the
Massive Stock Market API.

You have tools to check market status, get price snapshots, historical
price bars, company overviews, today's top gainers/losers, recent news,
dividend history, and a search tool over an ingested knowledge base of
company overviews and news.

Rules:
- Always use a tool to look up real data before answering questions about
  prices, performance, company facts, news, or dividends. Never invent
  numbers.
- For qualitative questions (what a company does, why it's in the news,
  general sentiment) prefer search_company_knowledge_base over
  get_company_overview/get_stock_news -- it searches ingested text with
  hybrid keyword+semantic search instead of returning raw API fields.
  Fall back to the live API tools if the knowledge base has nothing
  relevant (e.g. the ticker isn't on the ingested watchlist).
- If the user gives a company name instead of a ticker, infer the most
  likely ticker symbol yourself (e.g. "Apple" -> AAPL) before calling a tool.
- Prefer the smallest number of tool calls that answers the question, but
  make multiple calls (e.g. one per ticker) when the user compares several
  stocks.
- This tool data is not personalized investment advice. If asked for a
  recommendation ("should I buy X"), give the factual data and clearly
  note that you can't provide personal financial advice.
- Keep answers concise and cite the actual figures you retrieved (price,
  % change, date) rather than vague language.
- If a tool call returns an error (e.g. a date range or data type isn't
  available on the current API plan), do not keep retrying that tool with
  slightly different parameters. Try at most once more with an obviously
  different approach (e.g. a narrower or more recent date range); if that
  also fails, tell the user plainly what data isn't available instead of
  continuing to retry.
""".strip()

client = get_client()

_STRAY_CITATION_RE = re.compile("【[^【】]*】")


def _strip_stray_citations(text: str) -> str:
    """
    openai/gpt-oss-120b (Groq) sometimes echoes raw tool-call JSON back into
    its final answer, wrapped in CJK corner brackets 【...】 as a citation-
    style marker -- confirmed directly, e.g. a price answer that included
    '【{"ticker":"AAPL","bars":[...]}】' verbatim. Those brackets aren't used
    for anything else in normal English/markdown answers here, so stripping
    them (and tidying the resulting double space) is safe.
    """
    cleaned = _STRAY_CITATION_RE.sub("", text)
    return re.sub(r" {2,}", " ", cleaned).strip()


def agent_loop(
    question: str,
    previous_messages: list | None = None,
    model: str = MODEL,
    max_iterations: int = 5,
) -> tuple[AgentCallRecord, list]:
    """
    Run the function-calling agent loop for one user turn.

    Returns (record, messages):
    - record: an AgentCallRecord ready to hand to db_save.save_conversation()
    - messages: the full updated Responses-API message history, to pass
      back in as `previous_messages` for the next turn (multi-turn chat)
    """
    start = time.time()

    if previous_messages:
        messages = list(previous_messages)
        messages.append({"role": "user", "content": question})
    else:
        messages = [
            {"role": "developer", "content": INSTRUCTIONS},
            {"role": "user", "content": question},
        ]

    tool_calls_made = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    last_answer = ""

    for _ in range(max_iterations):
        response = responses_create_retry(
            client,
            model=model,
            input=messages,
            tools=TOOL_SCHEMAS,
        )

        total_prompt_tokens += response.usage.input_tokens
        total_completion_tokens += response.usage.output_tokens
        total_cost += calc_price(response.usage)["total_cost"]

        messages.extend(response.output)

        has_function_calls = False
        for item in response.output:
            if item.type == "function_call":
                has_function_calls = True
                args = json.loads(item.arguments)
                tool_calls_made.append({"name": item.name, "arguments": args})
                output_json = call_tool(item.name, args)
                messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": output_json,
                    }
                )
            elif item.type == "message":
                last_answer = _strip_stray_citations(item.content[0].text)

        if not has_function_calls:
            break

    record = AgentCallRecord(
        model=model,
        question=question,
        instructions=INSTRUCTIONS,
        answer=last_answer,
        tool_calls=tool_calls_made,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_prompt_tokens + total_completion_tokens,
        response_time=time.time() - start,
        cost=total_cost,
    )

    return record, messages


def build_toyaikit_runner(model: str = MODEL):
    """
    Same agent, built with the ToyAIKit teaching framework used across
    the course (Tools / OpenAIClient / OpenAIResponsesRunner), including
    04-evaluation's agent-evaluation lesson. Useful for interactive
    exploration -- see notebooks/explore_agent.ipynb.

    ToyAIKit's OpenAIClient calls client.responses.create()/.parse()
    internally (same Responses API surface Groq supports), and its
    constructor accepts a pre-built client -- so passing our
    Groq-configured `client` through is all that's needed to point it
    at Groq instead of OpenAI.
    """
    from toyaikit.chat import IPythonChatInterface
    from toyaikit.chat.runners import DisplayingRunnerCallback, OpenAIResponsesRunner
    from toyaikit.llm import OpenAIClient
    from toyaikit.tools import Tools

    from tools import ALL_FUNCTIONS

    agent_tools = Tools()
    for fn in ALL_FUNCTIONS:
        agent_tools.add_tool(fn)

    chat_interface = IPythonChatInterface()
    callback = DisplayingRunnerCallback(chat_interface)

    runner = OpenAIResponsesRunner(
        tools=agent_tools,
        developer_prompt=INSTRUCTIONS,
        chat_interface=chat_interface,
        llm_client=OpenAIClient(model=model, client=client),
    )

    return runner, callback


def extract_tool_calls(messages: list) -> list:
    """
    Pull {name, arguments} tool calls out of a ToyAIKit runner's
    `all_messages`, exactly as in 04-evaluation/lessons/14-agent-evaluation.md.
    Used by eval/evaluate.py when running the ToyAIKit-based agent.
    """
    calls = []
    for message in messages:
        if isinstance(message, dict):
            continue
        if getattr(message, "type", None) == "function_call":
            calls.append({"name": message.name, "arguments": message.arguments})
    return calls
