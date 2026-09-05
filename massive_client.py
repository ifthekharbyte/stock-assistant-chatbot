"""
massive_client.py

Thin wrapper around the Massive (https://www.massive.com) Stock Market
REST API. This plays the same role that ingest.py plays in the
llm-zoomcamp course: it is the one place that knows how to talk to the
outside data source. Everything above this layer (tools.py, agent.py)
only ever sees plain Python dicts/lists.

Auth: Massive accepts either `?apiKey=...` or `Authorization: Bearer ...`.
We use the Authorization header, matching Massive's own recommendation
for production use.

Docs: https://massive.com/docs/rest/quickstart
"""

import os
import time
from datetime import date, timedelta

import requests

BASE_URL = "https://api.massive.com"

# The free tier enforces a low requests-per-minute limit; ingesting even a
# handful of tickers can hit it well before finishing. Retry a 429 a few
# times with backoff (honoring Retry-After if the API sends one) instead of
# giving up on the first hit.
MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 15


class MassiveAPIError(RuntimeError):
    """Raised when the Massive API returns a non-2xx response."""


class MassiveClient:
    def __init__(self, api_key: str | None = None, base_url: str = BASE_URL, timeout: int = 15):
        self.api_key = api_key or os.getenv("MASSIVE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "MASSIVE_API_KEY is not set. Get a free key at "
                "https://massive.com/dashboard/signup and put it in your .env file."
            )
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        for attempt in range(MAX_RETRIES + 1):
            resp = self.session.get(url, params=params or {}, timeout=self.timeout)
            if resp.status_code == 429 and attempt < MAX_RETRIES:
                wait = float(resp.headers.get("Retry-After", BASE_BACKOFF_SECONDS * (attempt + 1)))
                time.sleep(wait)
                continue
            if not resp.ok:
                raise MassiveAPIError(f"GET {path} -> {resp.status_code}: {resp.text[:300]}")
            return resp.json()

    # ---------------------------------------------------------------
    # Market-wide
    # ---------------------------------------------------------------

    def market_status(self) -> dict:
        """Current trading status for US markets (open/closed/pre/post)."""
        return self._get("/v1/marketstatus/now")

    def top_movers(self, direction: str = "gainers") -> dict:
        """Top 20 gainers or losers today. direction: 'gainers' or 'losers'."""
        if direction not in ("gainers", "losers"):
            raise ValueError("direction must be 'gainers' or 'losers'")
        return self._get(f"/v2/snapshot/locale/us/markets/stocks/{direction}")

    # ---------------------------------------------------------------
    # Single ticker
    # ---------------------------------------------------------------

    def ticker_snapshot(self, ticker: str) -> dict:
        """Latest trade, quote, and today's/yesterday's OHLC for one ticker."""
        ticker = ticker.upper()
        return self._get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}")

    def previous_close(self, ticker: str) -> dict:
        """Previous trading day's OHLC + volume for one ticker."""
        ticker = ticker.upper()
        return self._get(f"/v2/aggs/ticker/{ticker}/prev")

    def aggregates(
        self,
        ticker: str,
        multiplier: int = 1,
        timespan: str = "day",
        from_date: str | None = None,
        to_date: str | None = None,
        limit: int = 120,
    ) -> dict:
        """
        Historical OHLCV bars, e.g. multiplier=1, timespan='day' for daily bars.
        Defaults to the last 6 months if no date range is given.
        """
        ticker = ticker.upper()
        if not to_date:
            to_date = date.today().isoformat()
        if not from_date:
            from_date = (date.today() - timedelta(days=182)).isoformat()
        path = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        return self._get(path, params={"adjusted": "true", "sort": "asc", "limit": limit})

    def ticker_details(self, ticker: str) -> dict:
        """Company overview: name, exchange, market cap, industry, description..."""
        ticker = ticker.upper()
        return self._get(f"/v3/reference/tickers/{ticker}")

    def news(self, ticker: str | None = None, limit: int = 5) -> dict:
        """Recent news articles, optionally filtered to one ticker, with sentiment."""
        params = {"limit": limit, "order": "desc", "sort": "published_utc"}
        if ticker:
            params["ticker"] = ticker.upper()
        return self._get("/v2/reference/news", params=params)

    def dividends(self, ticker: str, limit: int = 5) -> dict:
        """Recent cash dividend history for a ticker."""
        ticker = ticker.upper()
        return self._get("/stocks/v1/dividends", params={"ticker": ticker, "limit": limit, "order": "desc"})
