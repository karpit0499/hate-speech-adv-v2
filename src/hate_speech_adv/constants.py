from __future__ import annotations

PROTOCOL_ID = "grouped_v1"
DATASET_ID = "tdavidson/hate_speech_offensive"
DATASET_REVISION = "adc5fb774614827695774f2dbe0ea8122f6a92b4"
EXPECTED_ROWS = 24_783

LABELS = ("hate_speech", "offensive", "neither")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}
CODE_TO_SPLIT = {value: key for key, value in SPLIT_CODES.items()}
TUNING_SPLITS = frozenset(("train", "validation"))

# These are the complete static inputs that can affect split construction,
# validation selection, artifact evaluation/export, or the one-shot opening.
# The public manifest records their semantic/byte provenance and the locked
# selection receipt binds the same files plus the generated public manifest.
PROTOCOL_SOURCE_PATHS = (
    ".gitignore",
    "pyproject.toml",
    "configs/data/grouped_v1.json",
    "configs/evaluation/frozen_state.json",
    "configs/evaluation/slices_v1.json",
    "configs/export/int8.json",
    "configs/train/experiment_matrix.json",
    "configs/train/smoke.json",
    "evaluation/PROTOCOL.md",
    "evaluation/audit_data.py",
    "evaluation/build_manifest.py",
    "evaluation/compare_artifacts.py",
    "evaluation/evaluate.py",
    "evaluation/evaluate_artifact.py",
    "evaluation/open_frozen_test.py",
    "evaluation/reveal_frozen_test.py",
    "evaluation/requirements.txt",
    "evaluation/validate_frozen_state.py",
    "evaluation/validate_manifest.py",
    "requirements/dev.txt",
    "requirements/export.txt",
    "requirements/protocol.txt",
    "requirements/train.txt",
    "src/hate_speech_adv/__init__.py",
    "src/hate_speech_adv/artifact_evaluation.py",
    "src/hate_speech_adv/baselines.py",
    "src/hate_speech_adv/constants.py",
    "src/hate_speech_adv/data.py",
    "src/hate_speech_adv/export_pipeline.py",
    "src/hate_speech_adv/frozen_state.py",
    "src/hate_speech_adv/grouping.py",
    "src/hate_speech_adv/integrity.py",
    "src/hate_speech_adv/manifest.py",
    "src/hate_speech_adv/metrics.py",
    "src/hate_speech_adv/protocol.py",
    "src/hate_speech_adv/selection.py",
    "src/hate_speech_adv/slices.py",
    "src/hate_speech_adv/training_plan.py",
    "src/hate_speech_adv/transformer_pipeline.py",
    "training/export_onnx.py",
    "training/run_baselines.py",
    "training/run_matrix.py",
    "training/select_model.py",
)

PROTOCOL_LOCK_PATHS = PROTOCOL_SOURCE_PATHS + (
    "evaluation/manifests/grouped_v1.json",
)
