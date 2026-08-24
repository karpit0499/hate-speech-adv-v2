# `grouped_v1` evaluation protocol

`grouped_v1` is the only protocol eligible for new model selection. The old
public-seed evaluator and its row-derived metric artifact were retired:
normalized duplicate groups could cross its partitions, and publishing the
split recipe made membership reconstructible.

## Cryptographically sealed data contract

- Source: `tdavidson/hate_speech_offensive` at immutable revision
  `adc5fb774614827695774f2dbe0ea8122f6a92b4`.
- Group: SHA-256 of `template_v1` normalization (NFKC, HTML unescape,
  case-folding, leading-retweet removal, URL/handle placeholders, and
  whitespace collapse).
- Split: approximately 80/10/10 using two-stage stratified grouped folds. All
  random states and fold choices derive from a private 256-bit secret.
- Public manifest: dataset/config/source semantic hashes, aggregate split and
  class supports, an HMAC for the external train/validation artifact, and an
  independent HMAC for the sealed test. It contains no row identifiers, text,
  labels/votes at row level, group membership, or split assignments.
- Private tuning artifact: a mode-0600, ignored file that physically contains
  only train and validation rows. Training and validation read this artifact
  directly; they do not load the full source dataset and cannot encounter a
  test example.

The split secret must be exactly 64 lowercase hexadecimal characters, held
outside the repository, and supplied through a protected environment secret in
automation. Never print it or put it in an argument log.

## Rebuild and validate the foundation

Create an isolated environment and install the pinned protocol dependencies:

```bash
python3.12 -m venv work/venvs/classifier
work/venvs/classifier/bin/pip install -r requirements/dev.txt
```

The executable training stack is pinned to Torch `2.13.0`, Transformers
`5.15.1`, PEFT `0.20.0`, and Accelerate `1.14.0`; ONNX/export dependencies are
also maintained in `requirements/export.txt`.

Generate the commitments and external tuning artifact with a protected secret:

```bash
work/venvs/classifier/bin/python evaluation/build_manifest.py \
  --split-secret-file /protected/path/grouped_v1_split_secret \
  --private-output /protected/path/grouped_v1_tv.json
```

The private output is required to be outside the repository and is written
mode 0600. No in-repository path is accepted. Validate tracked artifacts and
guards with:

```bash
work/venvs/classifier/bin/python evaluation/validate_manifest.py
work/venvs/classifier/bin/python training/run_matrix.py --smoke
work/venvs/classifier/bin/python -m pytest -q
```

The committed public-manifest payload SHA-256 is regenerated with the protocol
and recorded in `evaluation/manifests/grouped_v1.json`. Rebuilding byte-for-byte
requires the same protected secret and pinned dependencies.

## Validation-only baselines

The majority and TF-IDF word+character logistic baselines use seeds 13, 42,
and 73, 2,000 group-aware bootstrap resamples, and every predeclared ambiguity
and hard-negative slice:

```bash
work/venvs/classifier/bin/python training/run_baselines.py \
  --split-secret-file /protected/path/grouped_v1_split_secret \
  --tuning-artifact /protected/path/grouped_v1_tv.json --check
```

The complete validation-only metrics, intervals, slice reports, and
seed-variance fields are in `evaluation/results/grouped_v1_baselines.json`.

## Frozen 27-run selection

The matrix has nine candidates over seeds 13, 42, and 73. It compares pinned
current/Twitter RoBERTa encoders, LoRA/partial/full adaptation, four declared
losses, and adaptation-specific learning rates. The smoke path constructs a
tiny local RoBERTa, wraps it through PEFT, and takes a real
`transformers.Trainer` step while offline and with socket access denied.

Full execution is explicit and reads only the external tuning artifact:

```bash
work/venvs/classifier/bin/python training/run_matrix.py --execute \
  --split-secret-file /protected/path/grouped_v1_split_secret \
  --tuning-artifact /protected/path/grouped_v1_tv.json \
  --candidate current_lora_weighted_ce --seed 13 \
  --output-dir training/runs/current_lora_weighted_ce/13
```

Each run records the exact matrix byte/semantic hashes, pinned encoder, complete
hyperparameters, software versions, 2,000 grouped bootstrap resamples, slices,
and a hash of every standalone-checkpoint file. Checkpoints are model-only and
non-resumable; the transient Trainer checkpoint is removed after the standalone
checkpoint is saved. The production seed is predeclared as 42.
For bounded storage, run seeds 13 and 73 before seed 42 for each candidate.
Non-production checkpoints are deleted immediately; after each complete
candidate, only the current validation leader's standalone seed-42 checkpoint
is retained.

Selection refuses anything except all 27 unique declared candidate/seed runs.
It is deliberately two-stage:

1. Commit the protocol before training.
2. Store the 27 `run_result.json` documents under `evaluation/results/` and
   commit them (model files remain ignored).
3. Create an unlocked validation selection draft from those 27 paths.
4. Lock the draft with both the earlier protocol commit and the results commit:

```bash
work/venvs/classifier/bin/python training/select_model.py \
  --lock-draft evaluation/selection/grouped_v1_selection_draft.json \
  --protocol-commit FULL_PROTOCOL_COMMIT_SHA \
  --results-commit FULL_RESULTS_COMMIT_SHA \
  --output evaluation/selection/grouped_v1_selection.json
```

The lock verifies every protocol and result blob from Git, not merely a SHA
written into JSON. It locks all opening/evaluation/report executables and their
required configs, requires protocol -> results ancestry, and rejects orphaned
objects. Commit the resulting locked receipt before any test opening. At open
time, results -> receipt/`HEAD` and protocol -> `HEAD` ancestry are required,
and the current bytes of every locked protocol file must still exactly match
the locked commit.

## Export and q8 validation gate

Export accepts only the exact seed-42 checkpoint tree named in the locked
receipt, requires a new empty output directory, enforces max length 256 and
ONNX opset 17, and hashes every ONNX/tokenizer/config output. The q8 comparison
requires fp32 and q8 ONNX evaluations bound to those exact output hashes and to
the exact export manifest/configuration.

```bash
work/venvs/classifier/bin/python training/export_onnx.py \
  --checkpoint /private/path/selected_checkpoint \
  --selection-receipt evaluation/selection/grouped_v1_selection.json \
  --output-dir /private/path/exported
```

Artifact evaluation and comparison remain validation-only and require the
external tuning artifact plus split secret. No q8 test result may influence the
validation gate.

## One-way frozen-test opening

The test loader requires all of the following: the secret, the exact committed
receipt at `HEAD`, verified protocol and result commit locks, the explicit
acknowledgement phrase, and a new ledger path outside the repository. It creates
the ledger with `O_EXCL`, mode 0600, and file/directory `fsync` before reading
the manifest, receipt, config, secret, dataset, or partition. An existing ledger
always fails, including after a mismatch, crash, or interrupted attempt.

```bash
work/venvs/classifier/bin/python evaluation/open_frozen_test.py \
  --receipt evaluation/selection/grouped_v1_selection.json \
  --split-secret-file /protected/path/grouped_v1_split_secret \
  --ledger /protected/path/grouped_v1_opening_ledger.json \
  --private-output /protected/path/grouped_v1_opened_test.json \
  --acknowledgement I_ACKNOWLEDGE_GROUPED_V1_TEST_IS_ONE_WAY
```

No frozen-test model score exists at this protocol stage. Do not run this
command until selection and q8 validation are final. After opening, report all
predeclared metrics, at least 1,000 grouped bootstrap resamples, all slices, and
all seed outcomes—never a best-seed-only result.

## Post-test state and optional reveal

`evaluation/validate_frozen_state.py` deliberately accepts three explicit
states: `pre_test_sealed`, `post_test_unrevealed`, and `post_test_revealed`.
Therefore CI passes before the one-shot run and does not require a premature
secret reveal. A completed frozen-test result must bind the public manifest,
selection receipt, opened-test payload, one-shot ledger, prediction records,
artifact results, and metrics using SHA-256 fields defined by
`hate_speech_adv.frozen_state`.

Only after that completed result exists may the split secret be turned into a
publishable reveal artifact:

```bash
work/venvs/classifier/bin/python evaluation/reveal_frozen_test.py \
  --receipt evaluation/selection/grouped_v1_selection.json \
  --ledger /protected/path/grouped_v1_opening_ledger.json \
  --opened-test /protected/path/grouped_v1_opened_test.json \
  --test-result evaluation/results/grouped_v1_frozen_test.json \
  --split-secret-file /protected/path/grouped_v1_split_secret
work/venvs/classifier/bin/python evaluation/validate_frozen_state.py
```

The reveal command refuses mismatched inputs, recomputes the sealed-test HMAC
from the already-opened rows, and binds the reveal to the completed test-result
payload. It is an available post-test action, not part of the current sealed
pre-test state; do not run it before the one-shot result is final.
