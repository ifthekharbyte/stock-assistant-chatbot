# Running ingestion on a schedule with Kestra

This sets up Kestra to run `ingest_kb.py` on a cron schedule instead of
by hand. **I haven't run this against a live Kestra instance** (no
Docker in the environment I built it in) — treat it as a documented,
best-effort starting point, not a verified recipe. If something doesn't
match what you see in the Kestra UI, Kestra's own docs
(https://kestra.io/docs) are the source of truth over this file.

## 1. Start your app's Postgres, on a named network

Kestra's task needs to reach the same Postgres the chat app uses. From
the project root:

```bash
make network    # docker network create monitoring
make postgres   # starts stock-assistant-pg on that network
make init-db    # creates the conversations/feedback/documents tables
```

## 2. Start Kestra

```bash
cd kestra
export SECRET_MASSIVE_API_KEY=$(echo -n "your-massive-api-key" | base64)
export SECRET_POSTGRES_PASSWORD=$(echo -n "password" | base64)
docker compose up -d
```

Open the UI at http://localhost:8080 — login `admin@kestra.io` / `Admin1234!`
(same defaults the course uses in `03-orchestration/lessons/03-setup.md`).

## 3. Get the project's Python files into the `stock-assistant` namespace

The flow (`flows/ingest_kb.yaml`) uses `namespaceFiles: enabled: true`,
which makes Kestra hand the task everything already uploaded to the
flow's namespace (`stock-assistant`) as its working directory. You need
to push these files there once (and again whenever they change):

- `ingest_kb.py`
- `db_documents.py`
- `db_init.py`
- `massive_client.py`
- `requirements-ingest.txt`
- `data/watchlist.json`

**Easiest way — the Kestra UI:** open the Editor, select the
`stock-assistant` namespace, and use the file browser's upload button to
add each file (keep `data/watchlist.json` under a `data/` folder to
match the path `ingest_kb.py` expects).

**Scriptable way — Kestra's namespace files API**, run from the project
root (verify the exact path against your Kestra version's API docs;
this is the pattern Kestra documents, not something I've run):

```bash
KESTRA_URL=http://localhost:8080
AUTH="admin@kestra.io:Admin1234!"
NS=stock-assistant

for f in ingest_kb.py db_documents.py db_init.py massive_client.py requirements-ingest.txt; do
  curl -s -u "$AUTH" -X PUT \
    "$KESTRA_URL/api/v1/namespaces/$NS/files?path=$f" \
    --data-binary "@$f"
done

curl -s -u "$AUTH" -X PUT \
  "$KESTRA_URL/api/v1/namespaces/$NS/files?path=data/watchlist.json" \
  --data-binary "@data/watchlist.json"
```

## 4. Import the flow

```bash
curl -X POST -u 'admin@kestra.io:Admin1234!' \
  http://localhost:8080/api/v1/flows/import \
  -F fileUpload=@flows/ingest_kb.yaml
```

Or paste `flows/ingest_kb.yaml` into the UI's flow editor directly.

## 5. Run it once manually before trusting the schedule

In the UI: namespace `stock-assistant` → flow `ingest_kb` → **Execute**.
Check the `run_ingestion` task's logs for how many documents it upserted,
and check `db_documents.py`'s `count_documents()` (or `SELECT COUNT(*)
FROM documents;`) against Postgres directly.

If the run fails, the most likely first issue is field names: `ingest_kb.py`
was written against Massive's *documented* response shape, not a live
response I've inspected (see the main README's verification notes) — you
may need to adjust `overview_document()`/`news_documents()` in
`ingest_kb.py` to match what Massive actually returns.

## 6. Turn on the schedule

Once a manual run succeeds, flip `disabled: true` to `disabled: false`
under the flow's `triggers:` section (edit in the UI, or edit
`flows/ingest_kb.yaml` and re-import) so it runs automatically on the
cron schedule (`0 11 * * 1-5` — 11:00 UTC on weekdays; adjust for your
timezone).
