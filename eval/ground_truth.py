"""
eval/ground_truth.py

Ground truth generation, following the structured-output pattern from
04-evaluation/lessons/02-ground-truth.md: instead of hand-writing test
questions, we ask an LLM to generate one per (tool, ticker) pair using
`responses.parse` + a Pydantic model, via evaluation_utils.llm_structured.

The course generates questions from FAQ *answers* and labels each with
the source document's id. We don't have fixed documents here -- our
"documents" are (tool, ticker) pairs -- so each generated question is
labeled with the tool it should require, playing the same role as the
course's `document` field in 04-evaluation/lessons/14-agent-evaluation.md
(there: "did search return the doc_id we generated the question from";
here: "did the agent call the tool we generated the question for").

Run with: python eval/ground_truth.py
Writes: eval/ground_truth.csv
"""

import csv
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel  # noqa: E402

from evaluation_utils import calc_total_price, get_client, llm_structured  # noqa: E402

client = get_client()

WATCHLIST_PATH = Path(__file__).resolve().parent.parent / "data" / "watchlist.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "ground_truth.csv"

TOOLS_TO_COVER = [
    "get_market_status",
    "get_stock_snapshot",
    "get_previous_close",
    "get_price_history",
    "get_company_overview",
    "get_top_movers",
    "get_stock_news",
    "get_dividends",
]

data_gen_instructions = """
You emulate a retail investor using a stock market chatbot. Given a tool
description and a ticker, write ONE natural-sounding question that a
real user would type, which requires that tool (for that ticker) to
answer correctly.

Write it the way people actually ask things online: not too formal, not
too short, not too long. Don't mention the tool name or use the raw
ticker symbol every time -- sometimes use the company name instead
(e.g. "Apple" instead of "AAPL").
""".strip()

TOOL_DESCRIPTIONS = {
    "get_market_status": "check whether US markets are open, closed, pre-market, or after-hours",
    "get_stock_snapshot": "get the latest price snapshot (last trade/quote, today's OHLC)",
    "get_previous_close": "get yesterday's open/high/low/close/volume",
    "get_price_history": "get historical price bars over a date range, for trend questions",
    "get_company_overview": "get company reference info: exchange, market cap, industry, description",
    "get_top_movers": "get today's top gaining or losing stocks",
    "get_stock_news": "get recent news articles with sentiment",
    "get_dividends": "get recent dividend payment history",
}


class GroundTruthQuestion(BaseModel):
    question: str


def generate_question(tool: str, ticker: str):
    user_prompt = json.dumps({"tool_purpose": TOOL_DESCRIPTIONS[tool], "ticker": ticker})
    result, usage = llm_structured(client, data_gen_instructions, user_prompt, GroundTruthQuestion)
    return result.question, usage


def main():
    watchlist = json.loads(WATCHLIST_PATH.read_text())
    tickers = [item["ticker"] for item in watchlist]

    rows = []
    usages = []
    for i, tool in enumerate(TOOLS_TO_COVER):
        ticker = tickers[i % len(tickers)]
        question, usage = generate_question(tool, ticker)
        usages.append(usage)
        rows.append({"tool": tool, "ticker": ticker, "question": question})
        print(f"[{tool}] {ticker}: {question}")

    with OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tool", "ticker", "question"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} ground-truth questions to {OUTPUT_PATH}")
    print(f"Generation cost: ${calc_total_price(usages):.5f}")


if __name__ == "__main__":
    main()
