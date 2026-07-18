FROM python:3.12-slim

WORKDIR /app
# Install CPU-only torch first (keeps the image small; no GPU needed for inference here)
RUN pip install --no-cache-dir "torch==2.4.1" --index-url https://download.pytorch.org/whl/cpu

# Everything else from PyPI, pinned in one place
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
ENV MODEL_PATH=kmrarpit2704/hs-roberta-v2
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]