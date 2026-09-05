# 📈 Stock Market Assistant

An agentic chatbot that answers questions about live and historical stock
market data by calling the [Massive Stock Market API](https://www.massive.com/stocks)
as a set of LLM tools. Built by following the structure and code patterns
taught in [DataTalksClub's LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp)
(2026 cohort) — specifically Module 1 (Agentic RAG / function calling),
Module 4 (Evaluation, including agent evaluation), Module 5 (Monitoring),
and the [project guidelines](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/project.md).

## Problem description

Retail investors want quick, conversational answers to questions like
*"How is Tesla doing today?"* or *"What's Apple's dividend history?"*
without hand-navigating a dashboard or a REST API. Static FAQ-style RAG
doesn't fit here — market data changes every second, so there is no fixed
corpus to index. Instead, this project uses an **agentic** pattern: the
LLM is given tools that call Massive's live REST endpoints, decides for
itself which tool(s) a question needs, reads the results, and answers
with real, current numbers instead of guessing.

## How it works (architecture)

```
ingest_kb.py (scheduled/semi-automated)
     │  pulls company overviews + news for the watchlist from Massive
     ▼
db_documents.py ──► Postgres `documents`  (the knowledge base)
     │
     ▼  loaded into memory at startup
knowledge_base.py ──► keyword index (minsearch) + vector index (ONNX embedder) ──► hybrid_search() [RRF]
     ▲
     │  tool call: search_company_knowledge_base
     │
User question
     │
     ▼
Streamlit chat (app.py)
     │
     ▼
Agent loop (agent.py)  ──tool call──►  tools.py  ──►  massive_client.py  ──►  Massive REST API (live data)
     │        ▲                              └────────►  knowledge_base.py  (ingested company/news data)
     │        └────────────────── tool result ◄─────────────────────────────────┘
     ▼
AgentCallRecord ──► db_save.py ──► Postgres `conversations`
     │
     ▼
judge.py (LLM-as-judge, runs on every turn) ──► db_feedback.py ──► Postgres `feedback` (source='judge')
     │
     ▼
User 👍/👎 in the UI ──► db_feedback.py ──► Postgres `feedback` (source='user')
     │
     ▼
monitoring/dashboard.py (Streamlit)  and/or  Grafana, both reading the same Postgres tables
```

The agent has two kinds of tools, and the system prompt tells it which to
prefer: live API tools (`get_stock_snapshot`, `get_price_history`, etc.)
for numbers that change by the second, and `search_company_knowledge_base`
— hybrid search over the ingested knowledge base — for qualitative
questions ("what does this company do", "why is it in the news").

This directly mirrors patterns from three course modules:

| Course file | This project | What it does |
|---|---|---|
| `01-agentic-rag/code/agents.ipynb` | `agent.py` | Manual function-calling loop: LLM decides which tool to call, reads results, repeats until it answers |
| `01-agentic-rag/code/rag_helper.py` | `agent.py` | Base assistant class/loop wrapping instructions + tools + model |
| `05-monitoring/code/metrics.py` (`LLMCallRecord`) | `agent.py` (`AgentCallRecord`) | Dataclass capturing model, prompt, answer, tokens, cost, response time per call |
| `05-monitoring/code/db_init.py` / `db_save.py` / `db_feedback.py` / `db_query.py` | Same file names, same functions | Postgres schema + read/write helpers for conversations and feedback |
| `05-monitoring/code/judge.py` | `judge.py` | Structured-output (`Literal["RELEVANT", ...]`) relevance judge, run automatically on every answer |
| `05-monitoring/code/app.py` | `app.py` | Streamlit chat that logs every turn, auto-judges it, and accepts 👍/👎 |
| `05-monitoring/code/dashboard.py` + `12-grafana.md` | `monitoring/dashboard.py` | Streamlit dashboard with the same panels the course builds in Grafana |
| `04-evaluation/code/evaluation_utils.py` | `evaluation_utils.py` | `llm_structured`, `llm_structured_retry`, `calc_price`, `map_progress` |
| `04-evaluation/lessons/02-ground-truth.md` | `eval/ground_truth.py` | LLM-generated test questions via structured output |
| `04-evaluation/lessons/14-agent-evaluation.md` (`AgentEvaluation`) | `eval/evaluate.py` | Two-dimensional judge: `answer_score` + `trajectory_score`, run in parallel with `ThreadPoolExecutor` |
| `01-agentic-rag/code/ingest.py` | `ingest_kb.py` + `db_documents.py` | Pulls source data (company overviews + news, instead of the FAQ) and builds a persistent knowledge base |
| `02-vector-search/lessons/02-embeddings.md` / `04-vector-search.md` | `knowledge_base.py` (`vector_search`) | `all-MiniLM-L6-v2` embeddings + numpy dot-product cosine similarity |
| `06-best-practices/lessons/02-hybrid-search.md` (`rrf()`) | `knowledge_base.py` (`rrf`, `hybrid_search`) | Reciprocal Rank Fusion combining keyword + vector rankings |
| `02-vector-search/lessons/09-onnx-embedder.md` | `embedder.py`, `download_embedding_model.py` | ONNX Runtime embeddings instead of sentence-transformers/PyTorch — same model, ~33x lighter dependency footprint |
| `07-project-example/lessons/02-evaluating-retrieval.md` (`hit_rate`, `mrr`) | `eval/search_evaluation.py` | Compares keyword vs. vector vs. hybrid retrieval, same metrics, same `evaluate()` shape |
| `03-orchestration/lessons/03-setup.md` (Kestra docker-compose, flow import via curl) | `kestra/docker-compose.yml`, `kestra/README.md` | Same standalone Kestra setup pattern, adapted to run a plain script instead of the AI-agent plugin |
| `01-agentic-rag/code/agents.ipynb` (ToyAIKit cells) | `agent.build_toyaikit_runner()` | ToyAIKit `Tools`/`OpenAIResponsesRunner`, used in `notebooks/explore_agent.ipynb` |
| `05-monitoring/lessons/13-docker-compose.md` | `Dockerfile` / `docker-compose.yaml` / `Makefile` | `uv`-based Dockerfile, Postgres + Grafana + app services |

One domain difference worth calling out: the course's FAQ agent has a
fixed *original answer* for every ground-truth question (it generates
questions from existing FAQ answers), so its judge can compare directly.
Live market data has no such fixed reference value, so — matching what
`13-llm-as-judge.md` itself says about production judging ("we usually
don't have that original answer for real user questions") — both the
live `judge.py` and the offline `eval/evaluate.py` judge relevance/quality
directly from the question and answer, not by diffing against a stored
"correct" answer.

## Data source: the Massive Stock Market API

[Massive](https://www.massive.com/stocks) provides REST endpoints for real-time
and historical US stock data (prices, fundamentals, news, corporate actions).
Auth is a bearer API key; `massive_client.py` wraps eight endpoints that the
agent can call as tools (`tools.py`):

| Tool | Massive endpoint |
|---|---|
| `get_market_status` | `GET /v1/marketstatus/now` |
| `get_stock_snapshot` | `GET /v2/snapshot/locale/us/markets/stocks/tickers/{ticker}` |
| `get_previous_close` | `GET /v2/aggs/ticker/{ticker}/prev` |
| `get_price_history` | `GET /v2/aggs/ticker/{ticker}/range/{mult}/{span}/{from}/{to}` |
| `get_company_overview` | `GET /v3/reference/tickers/{ticker}` |
| `get_top_movers` | `GET /v2/snapshot/locale/us/markets/stocks/{gainers\|losers}` — **not entitled on the free plan** (confirmed live: 403 "You are not entitled to this data") |
| `get_stock_news` | `GET /v2/reference/news` |
| `get_dividends` | `GET /stocks/v1/dividends` |

Get a free API key at [massive.com/dashboard/signup](https://massive.com/dashboard/signup) —
the Free ("Basic") plan covers snapshots, aggregates, reference data, and news, but has real
constraints confirmed directly against a live key (also documented at
[massive.com/pricing](https://massive.com/pricing)): **5 API calls/minute**, **2 years of
historical data** (older date-range requests 403), **end-of-day data freshness, not real-time**,
and `get_top_movers` (gainers/losers) requires a paid plan. `massive_client.py` retries 429s
with backoff honoring `Retry-After`; the agent's system prompt tells it not to keep retrying a
tool that fails for a plan-limit reason, but to explain the limitation to the user instead.

## Technologies

- **LLM**: [Groq](https://groq.com) (OpenAI-compatible Responses API + function calling), model `openai/gpt-oss-120b` by default. Swap the model via `MODEL_NAME` in `.env`. See "Using Groq" below.
- **Agent**: framework-free loop (`agent.py`) plus an optional [ToyAIKit](https://github.com/alexeygrigorev/toyaikit) runner for notebook exploration.
- **Data source**: Massive Stock Market API (live, no local knowledge base needed).
- **Interface**: [Streamlit](https://streamlit.io) chat app.
- **Database**: Postgres (`psycopg`), logging every conversation and both kinds of feedback.
- **Monitoring**: a built-in Streamlit dashboard, plus an optional Grafana service reading the same Postgres tables (`docker-compose.yaml`).
- **Evaluation**: LLM-generated ground truth + tool-selection hit rate + a two-dimensional LLM-as-judge (answer quality, trajectory quality) — see `eval/`.
- **Containerization**: `uv`-based Dockerfile + docker-compose (Postgres + Grafana + app + dashboard).

## Using Groq instead of OpenAI

This project runs on [Groq](https://console.groq.com) rather than OpenAI
— everywhere: the chat agent, the automatic judge, and both eval
scripts. This worked out simpler than a typical provider swap, and it's
worth explaining why, since the reasoning affects what you can trust
about it.

**Why this was a small change, not a rewrite.** Every LLM call in this
project uses OpenAI's newer Responses API
(`client.responses.create()`, `client.responses.parse(text_format=...)`)
rather than the older Chat Completions API. Groq has shipped a
Responses API that's directly compatible — same method names, same
`response.output` item shapes (`function_call`, `message`), same
`usage.input_tokens`/`output_tokens` fields. That meant the actual
agent loop in `agent.py`, the tool-call handling, and the structured-
output judge in `judge.py`/`evaluation_utils.py` needed **zero logic
changes** — only where the client points (`evaluation_utils.get_client()`
now builds an `OpenAI(base_url="https://api.groq.com/openai/v1", ...)`)
and which model gets asked for (`openai/gpt-oss-120b` by default,
chosen because Groq's own docs list it as supporting both custom
function/tool calling and `strict: true` structured outputs — the two
things this project actually needs).

**What I checked directly before making this switch**, rather than
assuming: Groq's Responses API docs (confirming `responses.create`/
`responses.parse` compatibility and the exact unsupported-features list),
their tool-use model support table (confirming `openai/gpt-oss-120b`
supports local/custom function calling, not just their built-in
web-search/code-execution tools), their structured-outputs docs
(confirming `openai/gpt-oss-120b` supports `strict: true` schema-
constrained output via the Responses API path our judge uses), and
current pricing for that model across five independent trackers, which
converged on $0.15 input / $0.60 output per 1M tokens — now what
`evaluation_utils.py`'s `PRICE_PER_1M_*` constants reflect.

**One documented Groq limitation, already handled:** Groq's Responses
API doesn't support stateful multi-turn conversations via
`previous_response_id` — you have to resend the full message history
every turn. `agent.py`'s loop already does exactly that (it never uses
`previous_response_id`), so this needed no change, but it's why the
loop is built the way it is.

**What I have *not* verified:** none of this has run against a real
Groq API key. The compatibility claims above come from reading Groq's
current documentation directly (not from training-data memory or
guesswork — same standard as everything else in this README's "What
I've verified" section), but "the docs say it works" and "I ran it and
it worked" are different confidence levels. If your first real run
surfaces a mismatch, the most likely places are: the exact
`response.output` item structure in edge cases, and whether
`openai/gpt-oss-120b`'s tool-calling quality holds up as well as
GPT-based models did for multi-step ticker comparisons — Groq only
serves open-weight models, and model *capability* (as opposed to API
*compatibility*) wasn't something I could test here.

```
stock-chat-assistant/
├── massive_client.py       # Massive API wrapper (the data-access layer)
├── ingest_kb.py              # Ingestion pipeline: overviews + news → documents table
├── knowledge_base.py          # keyword (minsearch) + vector (ONNX embedder) + hybrid (RRF) search
├── embedder.py                 # ONNX Runtime embedder (no PyTorch/CUDA) -- 02-vector-search/lessons/09-onnx-embedder.md
├── download_embedding_model.py  # one-time download of the ONNX model files
├── db_documents.py             # Postgres helpers for the `documents` table
├── tools.py                     # Tool functions + JSON schemas exposed to the LLM
├── agent.py                      # Agent loop (function-calling) + AgentCallRecord + ToyAIKit runner
├── judge.py                       # Automatic relevance judge, run on every turn
├── evaluation_utils.py             # llm_structured(_retry), calc_price, map_progress
├── db_init.py                       # Postgres connection + schema (conversations, feedback, documents)
├── db_save.py                        # Save conversations
├── db_feedback.py                     # Save feedback (judge + user, same table)
├── db_query.py                         # Read conversations/feedback for the dashboard
├── app.py                               # Streamlit chat interface
├── monitoring/
│   └── dashboard.py                      # Monitoring dashboard (5+ charts)
├── eval/
│   ├── ground_truth.py                    # Agent test questions (structured output)
│   ├── evaluate.py                         # Agent answer + trajectory evaluation
│   └── search_evaluation.py                 # Keyword vs. vector vs. hybrid retrieval evaluation
├── notebooks/
│   └── explore_agent.ipynb                   # Interactive walkthrough, mirrors agents.ipynb
├── tests/
│   ├── test_tools.py                          # Unit tests (mocked API)
│   └── test_knowledge_base.py                  # RRF + keyword/vector search unit tests
├── data/
│   └── watchlist.json                            # Tickers ingested into the knowledge base
├── pyproject.toml
├── requirements-ingest.txt   # minimal deps for the Kestra ingestion task
├── Dockerfile
├── docker-compose.yaml
├── Makefile
├── kestra/
│   ├── docker-compose.yml     # standalone Kestra, adapted from 03-orchestration
│   ├── README.md                # step-by-step setup for the scheduled flow
│   └── flows/
│       └── ingest_kb.yaml         # Schedule trigger + Python script task
└── .env.example
```

## Dependency footprint: what's necessary and what isn't

Every package in `pyproject.toml`'s main `dependencies` list is annotated
with which file actually imports it — audited by grepping every
top-level import across the whole codebase, not assumed. Two things
that were previously in the main dependency list turned out to be dead
weight in the running app and got moved to a separate `notebook` group
that's excluded by default:

| Package | Where it's actually used | Why it's not a runtime dependency |
|---|---|---|
| `jupyter` | Nowhere — it's a CLI tool to *open* `notebooks/explore_agent.ipynb`, not something any `.py` file imports | Pulls in ~60+ transitive packages (`ipykernel`, `jupyter-client`/`-core`, `ipython`, `pyzmq`, `tornado`, `debugpy`, and more — confirmed via `uv tree --package jupyter`), none of which the app, dashboard, or eval scripts touch |
| `toyaikit` | Only inside `agent.build_toyaikit_runner()`, which is a lazy import (`from toyaikit... ` sits inside the function body) called only from the notebook — `app.py`'s actual chat path uses `agent.agent_loop()`, which never touches it | Additionally pulls in the full Anthropic SDK and a pricing-lookup package, for a code path the running app never executes |

Moving both into a `notebook` dependency group (rather than deleting
them — the notebook is still a real, useful part of the project) and
having Docker install with `uv sync --no-dev` (which also skips the
`dev` group's `pytest`) cut the actual install from **153 packages down
to 68** — verified directly: resolved and installed both ways, counted
each with `uv pip list`, not estimated.

Combined with the earlier `torch`→ONNX Runtime swap (see "Using Groq"
section's sibling note in "Notes & limitations" below) and the missing
`.dockerignore` fix, the build should now be meaningfully smaller and
faster on every layer that was previously bloated — package count,
package weight per package, and build-context transfer size.

## Running it

### 1. Setup

```bash
cp .env.example .env
# fill in GROQ_API_KEY and MASSIVE_API_KEY in .env

uv sync
```

`uv sync` installs the app's runtime dependencies plus `pytest` (the
`dev` group, included by default). It does **not** install `jupyter` or
`toyaikit` — those are only needed for
`notebooks/explore_agent.ipynb` and live in a separate `notebook` group
that's excluded by default (see "Dependency footprint" below for why).
Run `uv sync --group notebook` if you want to open that notebook.

### 2. Download the embedding model (one-time)

```bash
make download-model
```

Downloads the ONNX embedding model (~87MB) used for the knowledge
base's vector/hybrid search into `models/`. Only needs to run once —
after that it's cached locally. Skip this if you only care about live
price/news tools and don't need `search_company_knowledge_base` to work.

### 3. Start Postgres and ingest the knowledge base

```bash
make postgres     # docker run postgres:17 on a local "monitoring" network
make init-db      # creates the conversations + feedback + documents tables
make ingest       # pulls company overviews + news for data/watchlist.json into `documents`
```

Re-run `make ingest` any time you want to refresh the knowledge base — it
upserts by document id, so re-running is safe and doesn't create
duplicates. In production, schedule it (cron, a GitHub Actions workflow,
or a Kestra flow — see "Scheduling ingestion" below) instead of running
it by hand.

### 4. Run the chat app

```bash
make chat         # streamlit run app.py
```

Open http://localhost:8501 and ask things like:

- "What's the current price of AAPL?"
- "Compare TSLA and RIVN over the last month"
- "Is the market open right now?"
- "What are today's top losers?"
- "Any recent news on NVDA?"
- "What's Coca-Cola's dividend history?"

Every answer is logged, auto-judged for relevance, and shows a 🟢/🟡/🔴
relevance badge plus 👍/👎 buttons for your own feedback.

### 5. Run everything with Docker

```bash
docker compose up --build
```

- Chat app: http://localhost:8501
- Monitoring dashboard: http://localhost:8502
- Grafana: http://localhost:3000 (login `admin`/`admin`) — point it at the
  `postgres` service and build the panels described in
  `05-monitoring/lessons/12-grafana.md` (response time, token usage, cost,
  tool usage, relevance distribution, user feedback, recent conversations)
  if you want alerting or more advanced visualization than the built-in
  Streamlit dashboard.

### 6. Run tests

```bash
make test
```

### 7. Run the evaluation

```bash
make eval
```

This generates `eval/ground_truth.csv` (one LLM-generated question per
tool), runs the agent on each, judges both the final answer and the
tool-call trajectory, and prints a tool-selection hit rate plus
answer/trajectory score breakdowns.

### 8. Run the search evaluation (keyword vs. vector vs. hybrid)

```bash
make search-eval
```

This generates two LLM-written questions per ingested document, then
computes Hit Rate and MRR for keyword search, vector search, and hybrid
(RRF) search over `knowledge_base.py`, and prints which one wins —
following `07-project-example/lessons/02-evaluating-retrieval.md`
exactly. If one method clearly beats the others, switch
`search_company_knowledge_base` in `tools.py` to call that method
directly instead of always using `hybrid_search`.

## Scheduling ingestion with Kestra

`kestra/` contains a Kestra flow (`flows/ingest_kb.yaml`) that runs
`ingest_kb.py` on a cron schedule via `io.kestra.plugin.scripts.python.Commands`
+ a `Schedule` trigger — unlike the course's own `03-orchestration/flows/`
examples, which are all built around Kestra's `io.kestra.plugin.ai.agent.AIAgent`
plugin for LLM-orchestration demos, not for running an external script.

See **`kestra/README.md`** for exact setup steps. Short version: start
Postgres (`make network && make postgres`), start Kestra
(`cd kestra && docker compose up -d`), push the project's `.py` files
into the `stock-assistant` namespace, import the flow, run it once
manually, then enable the schedule.

**I haven't run this against a live Kestra instance** (no Docker in the
environment I built it in) — the flow follows Kestra's documented
plugin conventions (the same `io.kestra.plugin.core.*` task types the
course's own flows use for logging, and the standard `scripts.python`
plugin for running arbitrary Python), but treat it as an unverified
starting point, not a tested recipe. If you'd rather not deal with that
risk, a cron job or scheduled GitHub Actions workflow calling
`uv run python ingest_kb.py` gets you the same schedule with far less
moving pieces — trade automation-tool sophistication for something
you can be more confident works.

## What I've verified vs. what I haven't

This section used to describe testing done without a live Massive API key,
Groq key, or running Postgres. It's since been run for real, end-to-end,
against a live Massive free-tier key, a live Groq key, and a real Postgres
container — and that run surfaced several real bugs, now fixed. Being
equally direct about this pass:

**Verified for real, end-to-end, this session:**
- `uv sync`, the ONNX model download, the full test suite (11/11), Postgres
  init (`db_init.py`), and `ingest_kb.py` all ran successfully against live
  services — ingestion pulled 48 real documents (8 overviews + 40 news
  articles) for the full watchlist into a real Postgres instance.
- `eval/search_evaluation.py` ran against that real ingested data: vector
  search won on both Hit Rate (1.00) and MRR (0.92) over keyword (0.97 /
  0.80) and hybrid (1.00 / 0.87).
- `eval/evaluate.py` (agent answer + trajectory evaluation) ran end-to-end
  against live Groq + live Massive calls: **100% tool-selection hit rate
  (8/8)**, 75% good answer score, 75% good trajectory score. The three
  "bad" scores are real and explainable, not app bugs: two are genuine
  Massive free-plan restrictions (a >2-year historical price request, and
  `get_top_movers` requiring a paid plan — both now documented above), and
  one is the agent making redundant duplicate tool calls on one question
  (an efficiency issue, not a correctness one).
- The full `app.py` chat flow — `agent_loop` → `save_conversation` →
  `evaluate_relevance` (judge) → `save_feedback` (both judge and user
  thumbs-up) → the dashboard's `get_conversations`/`get_stats`/
  `get_tool_usage`/`get_relevance_breakdown`/`get_user_feedback_breakdown`
  queries — was run end-to-end against real Postgres, and the Streamlit
  server itself was started and responded (200) on port 8501.
- `docker compose up --build` ran for real: both images built successfully
  (including the build-time ONNX model download), and all 4 containers
  (postgres, grafana, app, dashboard) started and stayed up, with the app
  and dashboard both returning HTTP 200 and clean startup logs.

**Real bugs found during this run and fixed:**
- **Missing `.env` loading in standalone entry points.** Only `agent.py`
  and `judge.py` called `load_dotenv()`; `ingest_kb.py`, `db_init.py`, and
  `evaluation_utils.py` (used by all of `eval/*.py`) didn't, so running them
  directly (as the README's own instructions say to) failed with
  "MASSIVE_API_KEY is not set" unless something else happened to import
  `agent.py` first. Fixed by adding `load_dotenv()` to all three. Also
  added `tests/conftest.py` so `pytest` loads `.env` before collecting
  `tests/test_tools.py` (which imports `tools.py`, which needs
  `MASSIVE_API_KEY` at import time).
- **No retry/backoff on Massive's 429s.** The free plan's 5 calls/minute
  limit is hit almost immediately ingesting more than ~2 tickers;
  `massive_client.py` now retries a 429 with backoff (honoring
  `Retry-After`) instead of silently skipping the document.
- **No retry/backoff on Groq's 429s**, and **oversized tool payloads.**
  `get_price_history`'s default 6-month range returned raw OHLCV bars with
  every Polygon-style field (15,453 JSON chars), and `get_stock_news`
  returned full publisher objects, image URLs, and a sentiment "insights"
  array for *every* ticker an article mentioned, not just the one asked
  about (26,830 chars for 5 JPM articles). Resent every turn (Groq's
  Responses API has no `previous_response_id`, so the full history is
  resent each iteration), either alone could exceed Groq's free-tier
  8,000-tokens-per-minute cap — confirmed directly via repeated live 429/413
  errors. Fixed by compacting both tools' output to just the fields a
  price/news question needs (`tools.py`'s `_compact_bars`/`_compact_news`),
  dropping `json.dumps(..., indent=2)` in favor of compact JSON, adding a
  shared `responses_create_retry()` (used by `agent.py`), tightening
  `agent_loop`'s `max_iterations` from 8 to 5, adding a system-prompt rule
  against retrying a failing tool more than once, and serializing
  `eval/evaluate.py`'s thread pools to `max_workers=1` — Groq's TPM cap is
  a *shared, organization-wide* budget, so two agent loops running
  concurrently can jointly exceed it even when neither is oversized alone
  (confirmed: a 413 at `max_workers=2` with no single call above ~6k
  tokens).
- **Raw tool JSON leaking into the final answer.** `openai/gpt-oss-120b`
  occasionally echoed a tool's raw JSON back into its answer wrapped in
  `【...】` citation-style brackets (e.g. `closed at **$222.38**
  【{"ticker":"AAPL",...}】`) — confirmed live, and present in 2 of the 8
  eval answers even though the judge scored them "good" anyway. Fixed with
  a small regex strip in `agent.py` (`_strip_stray_citations`).

**Still not verified:**
- The Kestra flow (`kestra/flows/ingest_kb.yaml`) has never run against a
  live Kestra instance — no Docker-based Kestra setup was exercised this
  session either. Treat `kestra/README.md` as a documented-conventions
  best effort, not a tested recipe.
- Real browser/UI testing (clicking through the Streamlit app, e.g. via
  Playwright or the `claude-in-chrome` extension) wasn't completed — a
  Playwright-driven check was interrupted by Groq's free-tier **daily**
  token cap (200,000 tokens/day) being exhausted by the rest of this
  session's live testing. The server itself was confirmed to start and
  respond, and the exact code path it calls (`agent_loop` →
  `save_conversation` → judge → `save_feedback` → dashboard queries) was
  verified directly, but the actual rendered page/click flow wasn't seen.

## Project rubric self-assessment

Scored against [project.md](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/project.md)'s
criteria, honestly, including where I'm not sure:

| Criterion | Estimate | Why |
|---|---|---|
| Problem description | 2/2 | Explained above without assuming course context. |
| Retrieval flow | 2/2 | Knowledge base (`documents`, populated by `ingest_kb.py`) + LLM — verified end-to-end against real ingested data. |
| Retrieval evaluation | 2/2 | `eval/search_evaluation.py` ran for real: vector search wins (Hit Rate 1.00, MRR 0.92) over keyword and hybrid. |
| LLM evaluation | 1/2 | `eval/evaluate.py` ran for real (100% tool-selection hit rate, 75% good answer/trajectory) but judges one model/one prompt; no comparison across models yet. |
| Interface | 2/2 | Streamlit chat UI; server confirmed to start and serve real requests, but a full click-through wasn't completed (see above). |
| Ingestion pipeline | 1–2/2* | `ingest_kb.py` ran for real against live Massive + Postgres (solid 1/2). The Kestra flow for the 2/2 bar is still unverified against a live Kestra instance — score 1/2 if you want the conservative read. |
| Monitoring | 2/2 | Feedback collection + dashboard queries verified end-to-end against real Postgres data this session. |
| Containerization | 2/2 | `docker compose up --build` ran for real this session: both images built, all 4 containers (postgres, grafana, app, dashboard) started and stayed up, and the app/dashboard both returned HTTP 200 with clean logs. |
| Reproducibility | 2/2 | Clear README, `.env.example`, pinned deps, `Makefile` — and the setup steps were just followed for real, successfully, end-to-end. |
| Best practices | 1/3 | Hybrid search implemented and evaluated (1 pt). No reranking beyond RRF's implicit fusion, no query rewriting. |
| Bonus | 0 | No cloud deployment. |

**Updated estimate: roughly 17–18/20 core points, 1/3 best-practice points.**
The remaining gaps are honest ones: LLM evaluation only covers one model,
query rewriting/reranking beyond RRF aren't implemented, the Kestra flow
remains unverified against a live Kestra instance, and there's no cloud
deployment yet.

## Notes & limitations

- **CPU-only by design, and lightweight about it.** The knowledge
  base's vector search originally used `sentence-transformers`, which
  depends on PyTorch — and PyPI's default PyTorch wheels bundle full
  CUDA/GPU support (several GB of `nvidia-*` packages) even on machines
  with no GPU. This project now uses the course's ONNX Runtime
  alternative instead (`embedder.py` + `download_embedding_model.py`,
  from `02-vector-search/lessons/09-onnx-embedder.md`): same model
  (`all-MiniLM-L6-v2`), same 384-dim output, same results — verified
  directly (see "What I've verified" above) — but via `onnxruntime` +
  `tokenizers` instead of `torch`. No GPU packages anywhere in the
  dependency tree (155 resolved packages, zero `torch`/`nvidia-*`), and
  the Docker image is dramatically smaller and faster to build as a
  result.
- The Dockerfile downloads the ONNX model at **build time**
  (`RUN python download_embedding_model.py`), so it's baked into the
  image and there's no runtime download on first request. If your build
  environment has no network access at build time, comment that line
  out of the `Dockerfile` and instead run
  `docker compose exec app python download_embedding_model.py` once
  after the container starts.
- `.dockerignore` keeps local `.venv/`, caches, and `.git/` out of the
  build context sent to Docker — without it, `docker build` can take
  minutes just transferring gigabytes of local files that were never
  needed in the image.

- Market data is only as fresh as your Massive plan allows; the free tier
  has some delay on certain endpoints.
- This assistant reports factual market data — it does not give personal
  investment advice, and the system prompt in `agent.py` enforces that.
- **Not covered here, but a natural extension**: Module 3 (Kestra
  orchestration) could schedule a recurring flow — e.g. a daily digest of
  top movers and watchlist prices sent to Slack/email — since that's a
  genuinely scheduled/batch job, unlike the on-demand chat flow this
  project focuses on.
