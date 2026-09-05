chat:
	uv run streamlit run app.py

dashboard:
	uv run streamlit run monitoring/dashboard.py --server.port=8502

download-model:
	uv run python download_embedding_model.py

network:
	docker network create monitoring

postgres: network
	docker run -it \
		--name stock-assistant-pg \
		--network monitoring \
		-e POSTGRES_USER=user \
		-e POSTGRES_PASSWORD=password \
		-e POSTGRES_DB=stock_assistant \
		-p 5432:5432 \
		-v pgdata:/var/lib/postgresql/data \
		postgres:17

init-db:
	uv run python db_init.py

query:
	uv run python db_query.py

test:
	uv run pytest

eval:
	uv run python eval/ground_truth.py
	uv run python eval/evaluate.py

ingest:
	uv run python ingest_kb.py

search-eval:
	uv run python eval/search_evaluation.py
