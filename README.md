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

The **Run evaluation** panel in the demo uses a small, hand-picked set of examples so you can watch the model be scored in real time. It is deliberately labelled *illustrative*. The model's real, headline performance numbers come from a proper held-out test set and are described in the [technical documentation](./TECHNICAL_DOCUMENTATION.md).

## Under the hood (one line)

A RoBERTa text-classification model, fine-tuned with LoRA on a public labelled dataset, exported to ONNX and served in-browser via transformers.js. The full pipeline (training, serving, deployment) is described in **[TECHNICAL_DOCUMENTATION.md](./TECHNICAL_DOCUMENTATION.md)**.

## License

Released under the [MIT License](./LICENSE).
