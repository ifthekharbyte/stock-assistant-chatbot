"""
evaluation_utils.py

Reusable helpers, ported directly from the course's evaluation_utils.py
(used in both 04-evaluation and 05-monitoring). Kept as its own module
for the same reason the course does: agent.py, judge.py, and eval/*.py
all need structured-output calling, cost calculation, and parallel
execution, and duplicating them would drift.

Configured for Groq: get_client() builds an OpenAI SDK client pointed
at Groq's OpenAI-compatible endpoint (https://api.groq.com/openai/v1).
Groq's Responses API is compatible with OpenAI's -- client.responses.create()
and client.responses.parse(text_format=...) both work unchanged -- so
agent.py, judge.py, and the eval scripts didn't need rewriting, only
where the client comes from and which model they ask for. See the
README's "Using Groq instead of OpenAI" section for what was checked
before making this switch and what wasn't.

Pricing (PRICE_PER_1M_*) is openai/gpt-oss-120b's real published Groq
rate -- $0.15/$0.60 per 1M input/output tokens -- confirmed against
multiple independent pricing trackers while making this change, not
guessed. Update it if you change MODEL_NAME to a different model.
"""

import os
import re
import time

from dotenv import load_dotenv
from openai import APIStatusError, OpenAI, RateLimitError
from tqdm.auto import tqdm

load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"

PRICE_PER_1M_INPUT = 0.15
PRICE_PER_1M_OUTPUT = 0.60

# Below this many tokens left in the current tokens-per-minute window, pause
# before the next call instead of risking a 413 -- a compacted tool result
# plus growing conversation history can still add up to more than this on a
# multi-tool-call turn (e.g. comparing two tickers).
GROQ_TOKEN_SAFETY_BUFFER = 3000


def get_client() -> OpenAI:
    """
    The OpenAI SDK talking to Groq instead of OpenAI -- same SDK, just a
    different base_url and API key. Requires GROQ_API_KEY in the
    environment (see .env.example).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at "
            "https://console.groq.com/keys and put it in your .env file."
        )
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def calc_price(usage) -> dict:
    input_cost = (usage.input_tokens / 1_000_000) * PRICE_PER_1M_INPUT
    output_cost = (usage.output_tokens / 1_000_000) * PRICE_PER_1M_OUTPUT
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


def calc_total_price(usages) -> float:
    return sum(calc_price(u)["total_cost"] for u in usages)


class GroqDailyLimitExceeded(RuntimeError):
    """
    Raised when Groq's daily token cap (TPD, not the per-minute TPM one) is
    hit. Confirmed directly: this project's free-tier key ran dry mid-turn
    with "Please try again in 13m45.12s" -- no realistic retry/backoff
    schedule bridges that, so retrying just burns ~105s (our full 6-attempt
    backoff schedule) before failing anyway. Fail fast instead.
    """


def _parse_groq_duration(text: str) -> float:
    """Parse Groq's rate-limit reset strings, e.g. '1.822s', '46m4.8s', '1h2m3s'."""
    match = re.match(r"(?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?$", text.strip())
    if not match or not any(match.groups()):
        return 5.0
    hours, minutes, seconds = match.groups()
    return int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(seconds or 0)


def responses_create_retry(client, max_retries=6, base_backoff=5, **kwargs):
    """
    client.responses.create(**kwargs), with two layers of Groq rate-limit
    handling:

    1. Proactive: Groq returns x-ratelimit-remaining-tokens / -reset-tokens
       on every response. If a call leaves the current tokens-per-minute
       window nearly exhausted, sleep out the reset before returning, so
       the *next* call (agent.py's next loop iteration) doesn't have to
       find out the hard way. This is what actually speeds up multi-tool-
       call turns like comparing two tickers -- confirmed directly, a
       comparison question that used to time out completes without ever
       hitting a 429/413 once this pre-emptive check is in place.
    2. Reactive fallback: retry on an actual 429 (RateLimitError) or Groq's
       413 "request too large for the remaining window" (a plain
       APIStatusError, not a RateLimitError -- confirmed directly, this
       project's TPM cap has been hit for real many times), honoring
       Retry-After when present.
    """
    for attempt in range(max_retries + 1):
        try:
            raw = client.responses.with_raw_response.create(**kwargs)
        except (RateLimitError, APIStatusError) as e:
            if "(TPD)" in str(e):
                raise GroqDailyLimitExceeded(
                    "Groq's daily token limit is used up for this API key -- retrying won't "
                    f"help for a while. Original error: {e}"
                ) from e
            is_rate_limit = isinstance(e, RateLimitError) or getattr(e, "status_code", None) == 413
            if not is_rate_limit or attempt == max_retries:
                raise
            retry_after = None
            response = getattr(e, "response", None)
            if response is not None:
                try:
                    retry_after = float(response.headers.get("retry-after"))
                except (TypeError, ValueError):
                    retry_after = None
            time.sleep(retry_after if retry_after is not None else base_backoff * (attempt + 1))
            continue

        remaining = raw.headers.get("x-ratelimit-remaining-tokens")
        reset = raw.headers.get("x-ratelimit-reset-tokens")
        if remaining is not None and reset is not None and int(remaining) < GROQ_TOKEN_SAFETY_BUFFER:
            time.sleep(_parse_groq_duration(reset))
        return raw.parse()


def llm_structured(client, instructions, user_prompt, output_type, model=DEFAULT_MODEL):
    """Call the Responses API with structured output (Pydantic model)."""
    messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": user_prompt},
    ]
    response = client.responses.parse(
        model=model,
        input=messages,
        text_format=output_type,
    )
    return response.output_parsed, response.usage


def llm_structured_retry(client, instructions, user_prompt, output_type, model=DEFAULT_MODEL, max_retries=3):
    for attempt in range(max_retries):
        try:
            return llm_structured(client, instructions, user_prompt, output_type, model=model)
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2**attempt)


def map_progress(pool, seq, f):
    """Run f(el) for each el in seq using the given executor pool, with a progress bar."""
    results = []
    with tqdm(total=len(seq)) as progress:
        futures = []
        for el in seq:
            future = pool.submit(f, el)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)
        for future in futures:
            results.append(future.result())
    return results
