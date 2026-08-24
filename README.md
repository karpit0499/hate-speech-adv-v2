# Hate-speech classifier: browser demo and defensible evaluation

This repository contains a three-class text classifier (`hate_speech`,
`offensive`, `neither`), a fully local browser demo, a hardened FastAPI serving
path, and a leakage-resistant evaluation protocol.

**Live demo:** https://karpit0499.github.io/hate-speech-adv-v2/

The browser performs classification locally with Transformers.js and WebAssembly.
Text entered for classification is not sent to this repository's server. The demo
currently loads the pinned historical q8 artifact at revision
`7c7c7c63c53f0e8373e5171c85c7e4a1a9a7e267`; its evaluation status is explained
below.

## Evaluation status

`grouped_v1` is the only protocol eligible for new model claims. It groups
case/HTML/retweet/URL/handle-normalized duplicates before splitting, keeps the
test membership outside Git, and physically separates train/validation data from
the frozen test.

The source has 24,783 rows and 24,556 normalized template groups. The audit found
121 duplicate groups containing 348 rows, including 12 groups with conflicting
labels. It also found substantial annotator ambiguity: 7,301 rows are
non-unanimous. Grouping keeps **zero** raw, strict-normalized, or
template-normalized groups across partitions.

Current public evidence is validation-only:

| Fixed baseline | Validation macro-F1 | Hate recall | Hate precision |
|---|---:|---:|---:|
| majority class | 0.2909 | 0.0000 | 0.0000 |
| TF-IDF word+character logistic regression | 0.7320 | 0.4126 | 0.3665 |

These figures use three declared seeds, 2,000 group-aware bootstrap resamples, and
the ambiguity and hard-negative slices in
[`evaluation/results/grouped_v1_baselines.json`](./evaluation/results/grouped_v1_baselines.json).
The frozen test has not been used for model inference or model selection at this
stage. The 27-run Transformer matrix, fixed-seed artifact export, fp32-to-q8 gate,
and one-way test opening are specified in
[`evaluation/PROTOCOL.md`](./evaluation/PROTOCOL.md).

### Historical artifact — not eligible

The deployed browser model came from a seed-42 row-stratified workflow that did not
group normalized duplicates: 57 normalized groups crossed its partitions. Its old
test recipe and score file have been retired so they cannot be mistaken for current
evidence or used to reconstruct the superseded split. The artifact remains available
only to keep the browser demo operational until a `grouped_v1` winner passes export,
q8 validation, frozen testing, and publication.

## Browser runtime evidence

The checked Chrome benchmark measures the pinned historical q8 artifact in the
actual JavaScript/WebAssembly path, not native ONNX Runtime:

| Input length (tokens) | Warm p50 | Warm p95 | Queries |
|---:|---:|---:|---:|
| 10 | 41.00 ms | 41.70 ms | 30 |
| 34 | 118.00 ms | 118.85 ms | 30 |
| 130 | 439.75 ms | 446.67 ms | 30 |
| 258 | 894.45 ms | 897.32 ms | 30 |

The first pipeline initialization observed in that run was 13.61 seconds and
included a 125,528,915-byte q8 model transfer. Browser caches were not cleared, so
this is not a guaranteed cold-load measurement. The recorded ~2.57 MB heap change
covers JavaScript heap only; it is not total browser or WASM memory. Raw timings,
environment details, failures, and byte counts are in
[`benchmarks/browser/runtime_chrome_151_macos_2026-08-20.json`](./benchmarks/browser/runtime_chrome_151_macos_2026-08-20.json).

## Reproduce the sealed protocol foundation

Use Python 3.12 and keep the 64-hex-character split secret outside the repository:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements/dev.txt
.venv/bin/python evaluation/build_manifest.py \
  --private-output /protected/path/grouped_v1_train_validation.json \
  --split-secret-file /protected/path/grouped_v1_split_secret
.venv/bin/python evaluation/validate_manifest.py
.venv/bin/python training/run_matrix.py --smoke
.venv/bin/python -m pytest -q
```

The private train/validation artifact is mode `0600` and ignored. Public manifests
contain commitments and aggregate support only—no row IDs, text, labels, or split
assignments. Opening the test requires a committed full-matrix selection receipt,
the protected secret, an explicit acknowledgement, and a durable create-once
ledger written before any source-data access.

## Serving and security

- The browser path is local-only and has no rationale API or remote LLM fallback.
- The FastAPI service validates bounded input, redacts raw text by default, and
  loads a revision-pinned model.
- The container runs as a non-root user; the Kubernetes example uses a read-only
  root filesystem, drops Linux capabilities, disables service-account token
  mounting, and declares resource limits.
- GitHub Actions use least permissions and immutable action SHAs. Dependabot,
  dependency review, CodeQL, secret scanning, and push protection are enabled.

## Limitations

- Dataset labels are crowd annotations and are frequently disputed, especially in
  the hate-speech class. Slice results must be read alongside aggregate metrics.
- The current live demo is the historical q8 artifact; a new grouped-protocol
  winner is not represented there until its export gate and publication complete.
- This classifier is a research/portfolio system, not a moderation policy or a
  substitute for human review. Errors can create real harm.

See [`TECHNICAL_DOCUMENTATION.md`](./TECHNICAL_DOCUMENTATION.md) for component
contracts and [`evaluation/PROTOCOL.md`](./evaluation/PROTOCOL.md) for the frozen
evaluation procedure.

## License

Released under the [MIT License](./LICENSE).
