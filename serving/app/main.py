import os, torch
import requests
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.sanitize import sanitize
from google.cloud import bigquery
import logging, uuid, datetime

MODEL_PATH = os.environ.get("MODEL_PATH")
if not MODEL_PATH:
    raise RuntimeError(
        "MODEL_PATH is not set. Point it at your Hugging Face repo id "
        "(e.g. your-username/hs-roberta-v2) — NOT the local 'hs-roberta-v2-merged' "
        "folder. Set it inline for local runs (see Phase 6.3) or via the Dockerfile "
        "ENV for the container (Phase 7.2)."
    )
ID2LABEL = {0: "hate_speech", 1: "offensive", 2: "neither"}

app = FastAPI(title="hate-speech-adv v2 classifier")

# Load once at startup (not per request)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

class ClassifyIn(BaseModel):
    text: str = Field(..., max_length=10000)   # hard cap; see Phase 11

class ClassifyOut(BaseModel):
    label: str
    confidence: float
    target_groups: list[str]
    rationale: str

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

LITELLM_URL = os.environ.get("LITELLM_URL")   # e.g. http://litellm:4000

def get_rationale(text: str, label: str) -> str:
    if not LITELLM_URL:
        return f"Model classified this text as '{label}'."
    try:
        r = requests.post(
            f"{LITELLM_URL}/v1/chat/completions",
            json={
                "model": "rationale-model",
                "messages": [{
                    "role": "user",
                    "content": (
                        f"A classifier already decided this text is '{label}'. "
                        f"In ONE sentence, explain why that label fits. "
                        f"Do not change or question the label.\n\nTEXT: {text}"
                    ),
                }],
            },
            timeout=5,
        )
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return f"Model classified this text as '{label}'."

log = logging.getLogger("uvicorn.error")
BQ_TABLE = os.environ.get("BQ_TABLE")   # e.g. "hate-speech-adv.moderation.classifications_raw"
bq = bigquery.Client() if BQ_TABLE else None

def write_row(text, out):
    if not bq:
        return
    errors = bq.insert_rows_json(BQ_TABLE, [{
        "message_id": str(uuid.uuid4()),
        "input_text": text,
        "label": out.label,
        "confidence": out.confidence,
        "target_groups": ",".join(out.target_groups),
        "rationale": out.rationale,
        "embedding": None,
        "model_version": "hs-roberta-v2",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }])
    if errors:                       # insert_rows_json RETURNS errors, it does not raise
        log.error("BigQuery insert failed: %s", errors)

@app.post("/classify", response_model=ClassifyOut)
def classify(body: ClassifyIn, background_tasks: BackgroundTasks):
    text = sanitize(body.text)
    inputs = tokenizer(text, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    idx = int(torch.argmax(probs))
    label = ID2LABEL[idx]
    confidence = float(probs[idx])
    # v2 keeps the schema stable but derives fields honestly:
    rationale = get_rationale(text, label)
    result = ClassifyOut(
        label=label,
        confidence=round(confidence, 4),
        target_groups=[],        # optional enrichment added in Phase 8 note
        rationale=rationale,
    )
    background_tasks.add_task(write_row, text, result)
    return result   # bound to a name so Phase 9 can hand it to a background task