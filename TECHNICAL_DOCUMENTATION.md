# Technical documentation

This document describes the current repository contracts. Historical row-stratified
metrics and the currently deployed browser artifact are retained for traceability,
but only the `grouped_v1` protocol is eligible for new model-quality claims.

## 1. Component boundaries

| Component | Purpose | Evidence boundary |
|---|---|---|
| `evaluation/`, `src/hate_speech_adv/` | split construction, metrics, slices, receipts, seals | public files contain commitments and aggregate support only |
| `training/` | baselines, 27-run matrix, selection, ONNX export | train/validation-only until explicit one-way opening |
| `docs/` | local Transformers.js/WebAssembly demo | currently serves the pinned historical q8 model |
| `serving/` | revision-pinned FastAPI inference | classification only; no remote rationale model |
| `k8s/` | example deployment | hardened pod defaults, not a managed production service |
| `benchmarks/browser/` | real-browser runtime evidence | JavaScript/WASM only; never relabelled native ONNX latency |

## 2. Data and labels

The source is `tdavidson/hate_speech_offensive` at immutable revision
`adc5fb774614827695774f2dbe0ea8122f6a92b4`. It contains 24,783 rows with three
labels: `hate_speech`, `offensive`, and `neither`.

Labels are crowd-vote outcomes, not objective ground truth. The audit records 7,301
non-unanimous rows; 1,167 of the 1,430 hate-speech rows are non-unanimous. Metrics
therefore include ambiguity slices and should not be interpreted as a complete
moderation policy.

### 2.1 Duplicate grouping

`template_v1` applies Unicode NFKC normalization, HTML unescaping, case folding,
leading-retweet removal, URL and handle placeholders, and whitespace collapse.
Rows sharing the resulting hash stay in the same partition.

The audit found:

- 121 template duplicate groups containing 348 rows;
- 12 template groups with conflicting labels, covering 51 rows;
- zero raw, strict-normalized, or template-normalized groups crossing partitions.

The historical seed-42 row split allowed 57 normalized groups to cross partitions,
so its checkpoints and scores are legacy evidence only.

### 2.2 Sealed split

Partitioning uses a private 256-bit secret and two-stage stratified grouped folds.
The tracked manifest contains data/config/source hashes, aggregate split support, and
HMAC commitments. It contains no row IDs, text, row labels/votes, group assignments,
or split membership.

Training reads an ignored mode-`0600` artifact that physically contains only train
and validation rows. The frozen-test opener claims a create-once ledger, flushes it
durably, and only then loads source data or reconstructs membership. A crash, hash
mismatch, or failed evaluation consumes the attempt.

See [`evaluation/PROTOCOL.md`](./evaluation/PROTOCOL.md) for commands and the exact
state machine.

## 3. Validation and model selection

### 3.1 Baselines

The fixed baselines are majority class and a TF-IDF word+character logistic model.
Both run at seeds 13, 42, and 73 and report 2,000 group-aware bootstrap resamples,
class metrics, ambiguity slices, and hard-negative slices. The frozen test is not
loaded.

### 3.2 Transformer matrix

The declared matrix has nine candidate configurations across seeds 13, 42, and 73:
27 runs in total. It covers two pinned RoBERTa-family encoders, LoRA/partial/full
adaptation, four declared loss families, and adaptation-specific learning rates.
Candidate selection uses validation evidence across all declared runs. Seed 42 is the
predeclared production/export seed; no best-seed checkpoint may be substituted.

Every run records:

- exact encoder revision, matrix/config/source hashes, hyperparameters, and software;
- aggregate, per-class, slice, and group-bootstrap validation results;
- an inventory and SHA-256 hash for every file in the standalone checkpoint;
- non-resumable checkpoint provenance.

Run directories must be new and empty. Trainer retains at most one transient model
checkpoint and removes it only after the exact standalone checkpoint is saved and
inventoried.

### 3.3 Commit locks

Selection is intentionally staged:

1. merge the protocol and executable source;
2. run and commit all 27 validation result documents;
3. lock the draft to the earlier protocol commit and the results commit;
4. commit the locked receipt before export or frozen-test access.

The lock verifies Git blobs, the declared source tree, current working-tree bytes,
and required commit ancestry. Merely naming a locally resolvable or orphan Git object
does not satisfy the gate.

## 4. Export and quantization

Export accepts only the selected seed-42 standalone checkpoint named by the committed
receipt. The contract fixes maximum sequence length 256 and ONNX opset 17, requires a
new output directory, and inventories every model, tokenizer, config, and ONNX file.

The validation-only artifact gate evaluates fp32 and dynamic-int8 ONNX outputs against
the same examples and binds the comparison to exact artifact hashes. Size reduction,
metric deltas, prediction agreement, slices, and runtime distributions are separate
fields. A small average delta cannot hide a failed hate-class or slice gate.

No new grouped-protocol q8 artifact is a publication candidate until this gate passes.
The live browser continues to use the historical pinned q8 revision in the meantime.

## 5. Frozen-test reporting

Test opening requires the protected secret, an exact committed receipt at the current
Git history, an explicit acknowledgement phrase, and a new ledger outside Git. The
opening/evaluation code itself is part of the frozen source lock.

The completed report must include all predeclared metrics, at least 1,000 grouped
bootstrap resamples, per-class results, all ambiguity and hard-negative slices, seed
variance, and both fp32/q8 artifact outcomes where declared. It may not trigger
retuning. A guarded post-test reveal can publish the partition only after the consumed
ledger and completed result are cryptographically linked.

## 6. Browser demo

`docs/index.html` uses Transformers.js 4.2.0 and WebAssembly. It pins the historical
q8 repository revision rather than following a mutable model branch. Classification
is local; there is no Groq key, Worker proxy, remote rationale request, or server-side
fallback in this path.

The checked Chrome 151 run measured 30 successful warm queries at each representative
length after three warmups:

| Tokens | p50 | p95 |
|---:|---:|---:|
| 10 | 41.00 ms | 41.70 ms |
| 34 | 118.00 ms | 118.85 ms |
| 130 | 439.75 ms | 446.67 ms |
| 258 | 894.45 ms | 897.32 ms |

The observed first initialization was 13.61 seconds with browser caches not cleared.
The q8 model transfer was 125,528,915 bytes. JavaScript heap delta is reported only as
a narrow diagnostic; it is not total browser/WASM memory. Historical native ONNX
timings are not used as browser evidence or a current runtime claim.

## 7. FastAPI serving contract

The optional server path loads one tokenizer/model at startup from a pinned model
revision and uses maximum sequence length 256. `POST /classify` accepts a bounded text
field and returns label, confidence, an empty target-group list, and a deterministic
classification statement. It never forwards moderation text to a remote language
model.

BigQuery analytics are disabled unless `BQ_TABLE` is configured. Raw text remains
redacted unless an operator explicitly sets `STORE_RAW_TEXT=true`; such a deployment
must separately define consent, retention, access, and deletion controls. The example
service is not exposed by this repository and must be placed behind deployment-level
authentication and rate limiting if made public.

## 8. Container and Kubernetes posture

- Python and ML dependencies are pinned to patched executable versions.
- The container runs as UID/GID 10001 and stores the Hugging Face cache in its own
  writable home.
- The Kubernetes example disables service-account token mounting, requires non-root,
  uses `RuntimeDefault` seccomp, drops all capabilities, prevents privilege escalation,
  and uses a read-only root filesystem with explicit writable cache/tmp volumes.
- Startup/readiness probes and CPU/memory requests and limits are declared.

The example manifest uses a local image name. Image signing, admission policy, network
policy, ingress authentication, and durable analytics governance remain deployment
operator responsibilities.

## 9. CI and repository security

Pull requests run protocol/unit checks, a real local no-network Transformer training
smoke, serving unit tests, dependency review, and CodeQL. Release actions are pinned to
immutable commit SHAs and use least permissions. The main-branch image job publishes
immutable commit tags plus SBOM/provenance to GHCR after required checks pass.

Dependabot, vulnerability alerts, automated security fixes, secret scanning, push
protection, and private vulnerability reporting are enabled. See
[`SECURITY.md`](./SECURITY.md) for reporting and supported-scope details.

## 10. Reproducible environments

Python 3.12 environments are separated by purpose:

- `requirements/protocol.txt` — split, baselines, metrics;
- `requirements/train.txt` — Torch 2.13.0, Transformers 5.15.1, PEFT 0.20.0,
  Accelerate 1.14.0, and training support;
- `requirements/export.txt` — isolated ONNX/export stack;
- `requirements/dev.txt` — protocol dependencies plus tests;
- `serving/requirements.txt` — server dependencies; the Dockerfile installs the
  patched CPU Torch wheel separately.

Do not install historical pins from old documentation or silently fall back to a
synthetic dataset/model when a pinned resource is unavailable.

## 11. Limitations

- The data reflects one English-language social-media dataset and its annotation
  process. Generalization, dialect fairness, and policy fit require separate evidence.
- Identity-term and profanity slices are diagnostic heuristics, not exhaustive fairness
  guarantees.
- The current browser artifact is historical until the grouped winner completes export,
  q8 validation, publication, and a new real-browser benchmark.
- A classifier score should support, not replace, contextual human moderation.

## License

MIT; see [`LICENSE`](./LICENSE).
