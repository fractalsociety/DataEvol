FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip \
    && pip install .

EXPOSE 8765

ENV DATAEVOL_API_TOKEN=dev-local-token

CMD ["dataevol", "serve", "--host", "0.0.0.0", "--port", "8765"]
