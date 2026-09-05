"""
ingest_kb.py

The ingestion pipeline: pulls company overviews and recent news for the
watchlist tickers (data/watchlist.json) from the Massive API and upserts
them into Postgres's `documents` table via db_documents.py.

This is what turns the project from "LLM + live API calls" into
"LLM + knowledge base" for the project rubric -- see the ingestion
pipeline and retrieval-flow criteria in project.md. It's a semi-
automated Python script (rubric: 1/2 points on its own); run it on a
schedule (cron, a GitHub Actions workflow, or a Kestra flow using
`io.kestra.plugin.scripts.python` with a `Schedule` trigger -- see the
"Scheduling ingestion" note in the README) to reach the 2/2 "automated
ingestion with a special tool" bar. I haven't built or tested a Kestra
flow for this -- see the README for why.

Two document types are ingested per ticker:
- "overview": one document, the company reference info (name, exchange,
  market cap, industry, description).
- "news": one document per recent news article, with title + summary.

Both become rows knowledge_base.py loads into memory to build the
keyword (minsearch) and vector (ONNX embedder) indexes.

Run with: python ingest_kb.py
Or via: make ingest
"""

import json

from dotenv import load_dotenv

load_dotenv()
import sys
from datetime import datetime, timezone
from pathlib import Path

from db_documents import upsert_documents
from db_init import init_db
from massive_client import MassiveClient

WATCHLIST_PATH = Path(__file__).resolve().parent / "data" / "watchlist.json"


def overview_document(client: MassiveClient, ticker: str) -> dict | None:
    try:
        data = client.ticker_details(ticker)
    except Exception as exc:  # noqa: BLE001
        print(f"  [overview:{ticker}] skipped: {exc}")
        return None

    result = data.get("results", data)
    name = result.get("name", ticker)
    description = result.get("description", "")
    industry = result.get("sic_description", "")
    exchange = result.get("primary_exchange", "")
    market_cap = result.get("market_cap", "")

    text = (
        f"{name} ({ticker}) trades on {exchange}. "
        f"Industry: {industry}. Market cap: {market_cap}. {description}"
    ).strip()

    return {
        "id": f"overview:{ticker}",
        "ticker": ticker,
        "doc_type": "overview",
        "title": f"{name} — Company Overview",
        "text": text,
        "source_url": None,
        "published_at": None,
    }


def news_documents(client: MassiveClient, ticker: str, limit: int = 5) -> list[dict]:
    try:
        data = client.news(ticker, limit=limit)
    except Exception as exc:  # noqa: BLE001
        print(f"  [news:{ticker}] skipped: {exc}")
        return []

    docs = []
    for article in data.get("results", []):
        article_id = article.get("id") or article.get("article_url", "")
        if not article_id:
            continue
        title = article.get("title", "")
        summary = article.get("description", "")
        published_at = article.get("published_utc")
        docs.append(
            {
                "id": f"news:{ticker}:{article_id}",
                "ticker": ticker,
                "doc_type": "news",
                "title": title,
                "text": f"{title}. {summary}".strip(),
                "source_url": article.get("article_url"),
                "published_at": published_at,
            }
        )
    return docs


def main():
    watchlist = json.loads(WATCHLIST_PATH.read_text())
    tickers = [item["ticker"] for item in watchlist]

    init_db()  # safe to call repeatedly; CREATE TABLE IF NOT EXISTS

    client = MassiveClient()

    all_docs = []
    for ticker in tickers:
        print(f"Ingesting {ticker}...")
        overview = overview_document(client, ticker)
        if overview:
            all_docs.append(overview)

        news = news_documents(client, ticker)
        all_docs.extend(news)
        print(f"  {len(news)} news article(s)")

    upsert_documents(all_docs)
    print(f"\nUpserted {len(all_docs)} documents into the knowledge base "
          f"at {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    sys.exit(main())
