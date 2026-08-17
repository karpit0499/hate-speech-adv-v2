# hate-speech-adv v2

A hate-speech text classifier that runs **entirely in your browser** — no sign-up, no server, nothing to install.

### 🔗 Try the live demo

**https://karpit0499.github.io/hate-speech-adv-v2/**

Type a sentence, click **Classify**, and the model tells you which of three categories it falls into. You can also click **Run evaluation** to watch the model score a set of example sentences live.

> The first classification takes a few seconds while the model file downloads to your browser (about 125 MB). Every classification after that is instant, because your browser keeps it cached.

---

## What it does

The model reads a short piece of text and sorts it into one of three categories:

| Category | Meaning |
|----------|---------|
| **hate speech** | Attacks or dehumanizes people because of who they are (e.g. religion, ethnicity, gender, sexual orientation, disability). |
| **offensive** | Rude, insulting, or crude, but **not** aimed at a group of people. |
| **neither** | Neutral, positive, or otherwise fine. |

Alongside the category, it shows a confidence score and a simple breakdown of how sure it was for each of the three options.

## How it works (in plain terms)

Most "AI moderation" tools send your text off to a large language model somewhere on the internet. This one is different: it's a small, specialized model that was **trained specifically for this three-way task**, and it's small enough to run right inside your web browser.

That has a few nice consequences:

- **Private** — your text never leaves your device during classification.
- **Free and always-on** — there's no backend to pay for or keep running.
- **Fast and consistent** — the same input always gives the same answer.

## A note on the numbers

The **Run evaluation** panel in the demo uses a small, hand-picked set of examples so you can watch the model be scored in real time. It is deliberately labelled *illustrative*. The model's headline performance numbers come from the stratified test split reproduced below.

## Reproduced held-out evaluation

The results below were reproduced from the published model files, not copied from a training log. The source dataset has 24,783 rows. It is split 80/20 with Hugging Face `datasets`, then that 20% holdout is split 50/50 into validation and test partitions. Both operations are stratified on the label and use `seed=42`. The final test partition has 2,479 examples: 143 `hate_speech`, 1,919 `offensive`, and 417 `neither`.

| Published artifact | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| fp32 PyTorch | 0.897943 | 0.744599 | 0.821070 | **0.775119** | 0.904605 |
| dynamic-int8 ONNX used by the browser demo | 0.895926 | 0.742197 | 0.814916 | **0.770863** | 0.902794 |

The full-precision per-class results are:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `hate_speech` | 0.420561 | 0.629371 | 0.504202 | 143 |
| `offensive` | 0.967421 | 0.912976 | 0.939410 | 1,919 |
| `neither` | 0.845815 | 0.920863 | 0.881745 | 417 |

Its confusion matrix (rows = true label, columns = predicted label, order = `hate_speech`, `offensive`, `neither`) is:

```text
[[  90,   38,  15],
 [ 112, 1752,  55],
 [  12,   21, 384]]
```

Dynamic int8 quantization cuts the ONNX file from 498,873,677 to 125,528,915 bytes: a **74.84% (3.97×) size reduction**. On an Apple M4 Pro (14-core CPU, 24 GB RAM), native ONNX Runtime 1.18.1 `CPUExecutionProvider` inference for the committed 14-token demo sentence measured **4.53 ms p50 / 5.56 ms p95** at batch size 1 after 25 warm-up runs over 500 measured runs. Including tokenization, it measured **4.86 ms p50 / 6.14 ms p95**. These are local native-runtime figures, not browser/WASM latency.

Exact dataset/model revisions, file hashes, the q8 per-class results, and the benchmark distribution are in [`evaluation/metrics.json`](./evaluation/metrics.json). Reproduce them with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r evaluation/requirements.txt
python evaluation/evaluate.py
```

## Under the hood (one line)

A RoBERTa text-classification model, fine-tuned with LoRA on a public labelled dataset, exported to ONNX and served in-browser via transformers.js. The full pipeline (training, serving, deployment) is described in **[TECHNICAL_DOCUMENTATION.md](./TECHNICAL_DOCUMENTATION.md)**.

## License

Released under the [MIT License](./LICENSE).
