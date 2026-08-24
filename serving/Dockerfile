FROM python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a

ARG APP_UID=10001
ARG APP_GID=10001

WORKDIR /app
# Install CPU-only torch first (keeps the image small; no GPU needed for inference here)
RUN pip install --no-cache-dir "torch==2.13.0" --index-url https://download.pytorch.org/whl/cpu

# Everything else from PyPI, pinned in one place
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN groupadd --gid "${APP_GID}" appuser \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home appuser \
    && mkdir -p /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /home/appuser

COPY --chown=appuser:appuser app/ ./app/
ENV MODEL_PATH=kmrarpit2704/hs-roberta-v2
ENV MODEL_REVISION=bcaa0d20b4dd0d7af95ae732d9007893328206ae
ENV HF_HOME=/home/appuser/.cache/huggingface
USER appuser:appuser
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
