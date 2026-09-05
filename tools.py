"""
tools.py

Defines the functions the agent is allowed to call, plus their JSON-schema
tool definitions. This mirrors the `search()` + `search_tool` pattern from
01-agentic-rag/code/agents.ipynb -- except instead of one `search` tool over
a static FAQ index, we expose several small tools over Massive's live
Stock Market API. The LLM decides which tool(s) to call and with what
arguments, the same way it decides when to call `search`.

Each function below:
- has a plain-English docstring (toyaikit's Tools.add_tool() reads this
  to auto-generate the JSON schema for the ToyAIKit-based runner), and
- is also registered manually in TOOL_SCHEMAS + TOOL_REGISTRY, so the
  raw agent_loop() in agent.py (no framework, just the OpenAI SDK) works
  identically to the notebook's manual `make_call()` pattern.
"""

import json

from massive_client import MassiveClient

client = MassiveClient()

_kb = None  # lazy-loaded singleton; see _get_kb()


def _get_kb():
    """
    Lazily load the knowledge base (built from data ingested by
    ingest_kb.py) on first use, so importing tools.py doesn't require a
    Postgres connection -- important for tests/test_tools.py, which
    mocks `client` but never touches the knowledge base.
    """
    global _kb
    if _kb is None:
        from knowledge_base import load_knowledge_base

        _kb = load_knowledge_base()
    return _kb


def get_market_status() -> dict:
    """
    Get whether US stock markets are currently open, closed, in
    pre-market, or in after-hours trading.
    """
    return client.market_status()


def get_stock_snapshot(ticker: str) -> dict:
    """
    Get the latest price snapshot for a single stock ticker: last trade,
    last quote, today's OHLC, and the previous day's close.

    Args:
        ticker: Stock ticker symbol, e.g. 'AAPL', 'TSLA', 'MSFT'.
    """
    return client.ticker_snapshot(ticker)


def get_previous_close(ticker: str) -> dict:
    """
    Get the previous trading day's open, high, low, close, and volume
    for a single stock ticker.

    Args:
        ticker: Stock ticker symbol, e.g. 'AAPL'.
    """
    return client.previous_close(ticker)


MAX_PRICE_HISTORY_BARS = 30


def _compact_bars(bars: list, max_bars: int = MAX_PRICE_HISTORY_BARS) -> list:
    """
    Trim raw Massive/Polygon-style OHLCV bars down to what a price/trend
    question actually needs: date + open/high/low/close/volume, dropping
    vwap/transaction-count noise, and evenly downsampling (always keeping
    the first and last bar) when the range has more than max_bars points.

    Long ranges (e.g. 6 months of daily bars) otherwise produce a JSON
    payload of ~15,000+ chars that, resent every turn (Groq's Responses
    API has no stateful previous_response_id), can single-handedly blow a
    free-tier tokens-per-minute budget -- confirmed directly: a 429
    "request too large" error on exactly this call, at 10,571 tokens
    against Groq's 8,000 TPM limit for openai/gpt-oss-120b.
    """
    trimmed = [{"t": b.get("t"), "o": b.get("o"), "h": b.get("h"), "l": b.get("l"), "c": b.get("c"), "v": b.get("v")} for b in bars]
    if len(trimmed) <= max_bars:
        return trimmed
    step = (len(trimmed) - 1) / (max_bars - 1)
    indices = sorted({round(i * step) for i in range(max_bars)})
    return [trimmed[i] for i in indices]


def get_price_history(
    ticker: str,
    multiplier: int = 1,
    timespan: str = "day",
    from_date: str = "",
    to_date: str = "",
) -> dict:
    """
    Get historical OHLCV price bars for a stock ticker over a date range.
    Useful for questions about trends, performance over time, or charting.

    Args:
        ticker: Stock ticker symbol, e.g. 'AAPL'.
        multiplier: Size of the timespan multiplier, e.g. 1 or 5.
        timespan: One of 'minute', 'hour', 'day', 'week', 'month'.
        from_date: Start date as YYYY-MM-DD. Defaults to 6 months ago if empty.
        to_date: End date as YYYY-MM-DD. Defaults to today if empty.
    """
    data = client.aggregates(
        ticker,
        multiplier=multiplier,
        timespan=timespan,
        from_date=from_date or None,
        to_date=to_date or None,
    )
    bars = data.get("results", [])
    result = {
        "ticker": data.get("ticker", ticker.upper()),
        "bar_count_returned_by_api": data.get("resultsCount", len(bars)),
        "bars": _compact_bars(bars),
    }
    if len(bars) > MAX_PRICE_HISTORY_BARS:
        result["note"] = (
            f"Downsampled from {len(bars)} to {len(result['bars'])} evenly-spaced bars to stay "
            "within token limits; each bar has date (t, ms epoch), open (o), high (h), low (l), "
            "close (c), volume (v)."
        )
    return result


def get_company_overview(ticker: str) -> dict:
    """
    Get company reference info for a stock ticker: name, exchange,
    market cap, industry (SIC), and a short description.

    Args:
        ticker: Stock ticker symbol, e.g. 'AAPL'.
    """
    return client.ticker_details(ticker)


def get_top_movers(direction: str = "gainers") -> dict:
    """
    Get today's top 20 gaining or losing stocks by percentage move.

    Args:
        direction: Either 'gainers' or 'losers'.
    """
    return client.top_movers(direction)


def _compact_news(articles: list, ticker: str | None) -> list:
    """
    Trim raw Massive/Polygon news articles down to what a headlines/sentiment
    question needs. The raw payload includes a full publisher object (name +
    3 URLs), image URLs, a keywords array, and a per-ticker "insights" array
    covering every ticker the article mentions -- not just the one asked
    about. Confirmed directly: 5 raw JPM articles came to 26,830 chars
    (mostly insights for ~10 unrelated tickers per article), and resending
    that every turn (Groq's Responses API has no previous_response_id) was
    enough on its own to exceed a free-tier 8,000 tokens-per-minute cap.
    """
    compact = []
    for a in articles:
        insights = a.get("insights") or []
        sentiment = None
        if ticker:
            match = next((i for i in insights if (i.get("ticker") or "").upper() == ticker.upper()), None)
            if match:
                sentiment = {"sentiment": match.get("sentiment"), "reasoning": match.get("sentiment_reasoning")}
        compact.append(
            {
                "title": a.get("title"),
                "publisher": (a.get("publisher") or {}).get("name"),
                "published_utc": a.get("published_utc"),
                "description": a.get("description"),
                "sentiment": sentiment,
            }
        )
    return compact


def get_stock_news(ticker: str = "", limit: int = 5) -> dict:
    """
    Get recent news articles with sentiment, optionally filtered to one
    ticker.

    Args:
        ticker: Stock ticker symbol to filter by. Leave empty for general market news.
        limit: Max number of articles to return (default 5).
    """
    data = client.news(ticker or None, limit=limit)
    return {"articles": _compact_news(data.get("results", []), ticker or None)}


def search_company_knowledge_base(query: str, ticker: str = "") -> dict:
    """
    Search the ingested knowledge base of company overviews and recent
    news (built by ingest_kb.py from the Massive API) using hybrid
    keyword + semantic search. Prefer this over get_company_overview or
    get_stock_news for qualitative questions ("what does this company
    do", "why is the stock in the news", "what's the sentiment around
    X") since it searches ingested text directly instead of returning
    raw API fields.

    Args:
        query: What to search for, in natural language.
        ticker: Restrict results to one ticker. Leave empty to search across the whole watchlist.
    """
    kb = _get_kb()
    results = kb.hybrid_search(query, ticker=ticker or None, num_results=5)
    return {"results": results}


def get_dividends(ticker: str, limit: int = 5) -> dict:
    """
    Get recent cash dividend payment history for a stock ticker.

    Args:
        ticker: Stock ticker symbol, e.g. 'KO'.
        limit: Max number of dividend records to return (default 5).
    """
    return client.dividends(ticker, limit=limit)


# ---------------------------------------------------------------------
# Manual registry, used by the framework-free agent_loop() in agent.py
# ---------------------------------------------------------------------

ALL_FUNCTIONS = [
    get_market_status,
    get_stock_snapshot,
    get_previous_close,
    get_price_history,
    get_company_overview,
    get_top_movers,
    get_stock_news,
    search_company_knowledge_base,
    get_dividends,
]

TOOL_REGISTRY = {fn.__name__: fn for fn in ALL_FUNCTIONS}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "get_market_status",
        "description": "Check whether US stock markets are open, closed, pre-market, or after-hours right now.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "get_stock_snapshot",
        "description": "Get the latest price snapshot (last trade, last quote, today's OHLC) for one stock ticker.",
        "parameters": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"}},
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_previous_close",
        "description": "Get the previous trading day's OHLC and volume for one stock ticker.",
        "parameters": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"}},
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_price_history",
        "description": "Get historical OHLCV price bars for a ticker over a date range, for trend/performance questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"},
                "multiplier": {"type": "integer", "description": "Bar size multiplier, e.g. 1 or 5"},
                "timespan": {"type": "string", "description": "minute, hour, day, week, or month"},
                "from_date": {"type": "string", "description": "YYYY-MM-DD, defaults to 6 months ago"},
                "to_date": {"type": "string", "description": "YYYY-MM-DD, defaults to today"},
            },
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_company_overview",
        "description": "Get company reference info for a ticker: name, exchange, market cap, industry, description.",
        "parameters": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"}},
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_top_movers",
        "description": "Get today's top 20 gaining or losing stocks by percentage.",
        "parameters": {
            "type": "object",
            "properties": {"direction": {"type": "string", "description": "'gainers' or 'losers'"}},
            "required": ["direction"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_stock_news",
        "description": "Get recent news articles with sentiment, optionally filtered to one ticker.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker to filter by, or empty for general market news"},
                "limit": {"type": "integer", "description": "Max articles to return"},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "search_company_knowledge_base",
        "description": (
            "Search ingested company overviews and news (hybrid keyword + semantic search) "
            "for qualitative questions -- what a company does, why it's in the news, sentiment, etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query"},
                "ticker": {"type": "string", "description": "Restrict to one ticker, or empty for the whole watchlist"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_dividends",
        "description": "Get recent cash dividend payment history for a ticker.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. KO"},
                "limit": {"type": "integer", "description": "Max records to return"},
            },
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
]


def call_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name and return its result as a JSON string."""
    if name not in TOOL_REGISTRY:
        return json.dumps({"error": f"Unknown tool '{name}'"})
    try:
        result = TOOL_REGISTRY[name](**arguments)
        return json.dumps(result, default=str)
    except Exception as exc:  # noqa: BLE001 -- surfaced to the LLM, not crashed on
        return json.dumps({"error": str(exc)})
