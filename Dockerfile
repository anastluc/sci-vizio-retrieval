# Sci-Vizio API and worker image (Heroku container stack; also usable locally).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    EMBEDDING_CACHE_DIR=/opt/models

WORKDIR /app

COPY requirements-saas.txt .
RUN pip install -r requirements-saas.txt

# Bake the embedding model into the image: dyno filesystems are ephemeral, so
# downloading at boot would slow every restart.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2', cache_dir='/opt/models')"

COPY config.ini .
COPY sci_vizio_retrieval sci_vizio_retrieval
COPY sci_vizio_saas sci_vizio_saas

RUN useradd --create-home app && chown -R app /app /opt/models
USER app

EXPOSE 8000
# Heroku sets $PORT; the proxy headers make request URLs use https.
CMD uvicorn sci_vizio_saas.main:create_app --factory --host 0.0.0.0 --port ${PORT:-8000} \
    --proxy-headers --forwarded-allow-ips='*'
