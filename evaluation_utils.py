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
import time

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from tqdm.auto import tqdm

load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"

PRICE_PER_1M_INPUT = 0.15
PRICE_PER_1M_OUTPUT = 0.60


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


def responses_create_retry(client, max_retries=6, base_backoff=5, **kwargs):
    """
    client.responses.create(**kwargs), retrying on Groq/OpenAI 429s.

    Groq's free tier enforces a low tokens-per-minute cap that's easy to
    hit with a full tool schema resent every turn (see agent.py) or a few
    concurrent eval workers (see eval/evaluate.py) -- without this, either
    surfaces the raw RateLimitError to the user/caller instead of just
    waiting out the window.
    """
    for attempt in range(max_retries + 1):
        try:
            return client.responses.create(**kwargs)
        except RateLimitError as e:
            if attempt == max_retries:
                raise
            retry_after = None
            response = getattr(e, "response", None)
            if response is not None:
                try:
                    retry_after = float(response.headers.get("retry-after"))
                except (TypeError, ValueError):
                    retry_after = None
            time.sleep(retry_after if retry_after is not None else base_backoff * (attempt + 1))


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
