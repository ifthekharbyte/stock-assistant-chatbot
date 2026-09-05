FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml .python-version ./
# --no-dev skips both the dev group (pytest) and the notebook group
# (jupyter, toyaikit) -- neither is imported anywhere in the actual
# running app. Cuts the install from 153 packages to 68; see
# pyproject.toml's comments for the full audit. Run `uv sync` (no flag)
# locally if you want jupyter/toyaikit for notebooks/explore_agent.ipynb.
RUN uv sync --no-dev

COPY . .

# Bake the ONNX embedding model into the image at build time instead of
# downloading it on first use at runtime -- see download_embedding_model.py
# and 02-vector-search/lessons/09-onnx-embedder.md. This needs network
# access during `docker build`; if that's not available in your build
# environment, run it after the container starts instead:
#   docker compose exec app python download_embedding_model.py
RUN python download_embedding_model.py

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
