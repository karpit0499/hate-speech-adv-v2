from __future__ import annotations

import inspect
import importlib.metadata
import json
import platform
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .constants import LABELS
from .integrity import artifact_tree, file_sha256, payload_sha256, semantic_file_sha256
from .integrity import verify_payload_integrity
from .manifest import (
    load_manifest,
    load_tuning_artifact,
    records_for_tuning,
    require_external_private_path,
)
from .metrics import classification_metrics, grouped_bootstrap_interval, hate_recall_value, macro_f1_value
from .protocol import read_split_secret
from .slices import slice_metric_report
from .training_plan import load_matrix, validate_matrix


def _candidate(matrix: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    matches = [candidate for candidate in matrix["candidates"] if candidate["id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"unknown candidate: {candidate_id}")
    return matches[0]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _class_weights(labels: list[int], strategy: str, beta: float = 0.9999) -> np.ndarray:
    counts = np.asarray([Counter(labels)[index] for index in range(len(LABELS))], dtype=np.float64)
    if np.any(counts == 0):
        raise ValueError("every class must occur in training")
    if strategy in ("weighted_ce", "focal"):
        weights = len(labels) / (len(LABELS) * counts)
    elif strategy == "class_balanced":
        weights = (1 - beta) / (1 - np.power(beta, counts))
        weights /= weights.mean()
    else:
        weights = np.ones(len(LABELS), dtype=np.float64)
    return weights.astype(np.float32)


def _apply_adaptation(model: Any, adaptation: str, settings: dict[str, Any]) -> Any:
    if adaptation == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return model
    if adaptation == "partial":
        for parameter in model.parameters():
            parameter.requires_grad = False
        classifier = getattr(model, "classifier", None)
        if classifier is None:
            raise ValueError("partial fine-tuning requires a classifier module")
        for parameter in classifier.parameters():
            parameter.requires_grad = True
        base = getattr(model, "roberta", None)
        if base is None or not hasattr(base, "encoder"):
            raise ValueError("partial fine-tuning currently requires a RoBERTa encoder")
        layers = list(base.encoder.layer)
        count = int(settings["unfrozen_encoder_layers"])
        if count < 1 or count > len(layers):
            raise ValueError("invalid partial unfreeze layer count")
        for layer in layers[-count:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        return model
    if adaptation == "lora":
        from peft import LoraConfig, TaskType, get_peft_model

        configuration = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=int(settings["rank"]),
            lora_alpha=int(settings["alpha"]),
            lora_dropout=float(settings["dropout"]),
            target_modules=list(settings["target_modules"]),
            modules_to_save=list(settings["modules_to_save"]),
        )
        return get_peft_model(model, configuration)
    raise ValueError(f"unsupported adaptation: {adaptation}")


def require_production_seed_last(
    matrix_path: str | Path,
    candidate_id: str,
    seed: int,
    output_dir: str | Path,
) -> Path:
    """Make seed 42 last so only a complete candidate can retain its weights."""
    matrix = load_matrix(matrix_path)
    validate_matrix(matrix)
    output = Path(output_dir).resolve()
    run_root = output.parent.parent
    if output.name != str(seed) or output.parent.name != candidate_id:
        raise ValueError("matrix outputs must use RUN_ROOT/<candidate>/<seed>")
    production_seed = int(matrix["selection"]["production_seed"])
    if seed == production_seed:
        missing = [
            other_seed
            for other_seed in matrix["seeds"]
            if other_seed != production_seed
            and not (output.parent / str(other_seed) / "run_result.json").is_file()
        ]
        if missing:
            raise ValueError(
                f"production seed {production_seed} must run last for bounded storage; "
                f"missing candidate results for seeds {missing}"
            )
    return run_root


def prune_matrix_checkpoint_storage(
    matrix_path: str | Path,
    run_root: str | Path,
) -> dict[str, Any]:
    """Keep only the current complete-matrix leader's production-seed weights."""
    matrix = load_matrix(matrix_path)
    validate_matrix(matrix)
    root = Path(run_root).resolve()
    candidate_ids = {str(candidate["id"]) for candidate in matrix["candidates"]}
    required_seeds = {int(seed) for seed in matrix["seeds"]}
    production_seed = int(matrix["selection"]["production_seed"])
    grouped: dict[str, dict[int, dict[str, Any]]] = {}
    for result_path in sorted(root.glob("*/*/run_result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        verify_payload_integrity(result)
        candidate_id = str(result.get("candidate_id", ""))
        seed = int(result.get("seed", -1))
        if candidate_id not in candidate_ids or seed not in required_seeds:
            raise ValueError(f"undeclared run result under matrix root: {result_path}")
        if result_path.parent != root / candidate_id / str(seed):
            raise ValueError(f"non-canonical run-result path: {result_path}")
        grouped.setdefault(candidate_id, {})[seed] = result

    minimum_precision = float(matrix["selection"]["minimum_hate_speech_precision"])
    complete = []
    for candidate_id, by_seed in grouped.items():
        if set(by_seed) != required_seeds:
            continue
        macro_mean = float(np.mean([run["metrics"]["macro_f1"] for run in by_seed.values()]))
        recall_mean = float(
            np.mean([run["metrics"]["hate_speech_recall"] for run in by_seed.values()])
        )
        precision_mean = float(
            np.mean([run["metrics"]["hate_speech_precision"] for run in by_seed.values()])
        )
        if precision_mean >= minimum_precision:
            complete.append((macro_mean, recall_mean, candidate_id))
    leader = max(complete, default=None)
    retained = (
        root / leader[2] / str(production_seed) / "selected_checkpoint"
        if leader is not None
        else None
    )
    pruned: list[str] = []
    for checkpoint in sorted(root.glob("*/*/selected_checkpoint")):
        resolved = checkpoint.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise PermissionError("checkpoint cleanup target escapes the run root") from error
        if retained is not None and resolved == retained.resolve():
            continue
        shutil.rmtree(resolved)
        pruned.append(str(resolved.relative_to(root)))
    remaining = sorted(root.glob("*/*/selected_checkpoint"))
    if len(remaining) > 1:
        raise RuntimeError("bounded checkpoint retention left more than one production candidate")
    return {
        "production_seed": production_seed,
        "leader_candidate": leader[2] if leader is not None else None,
        "retained_checkpoint": (
            str(remaining[0].relative_to(root)) if remaining else None
        ),
        "pruned_checkpoints": pruned,
        "maximum_retained_standalone_checkpoints": 1,
    }


def run_training(
    matrix_path: str | Path,
    manifest_path: str | Path,
    tuning_artifact_path: str | Path,
    split_secret_path: str | Path,
    slices_path: str | Path,
    candidate_id: str,
    seed: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Train one declared candidate on train and validation only."""
    import torch
    import torch.nn.functional as functional
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    matrix_path = Path(matrix_path)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("training output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = load_matrix(matrix_path)
    validate_matrix(matrix)
    if seed not in matrix["seeds"]:
        raise ValueError("seed is not in the frozen experiment matrix")
    candidate = _candidate(matrix, candidate_id)
    _set_seed(seed)
    from transformers import set_seed as set_transformers_seed

    set_transformers_seed(seed)
    manifest = load_manifest(manifest_path)
    tuning_artifact_path = require_external_private_path(
        tuning_artifact_path,
        Path(__file__).resolve().parents[2],
        "private train/validation artifact",
    )
    tuning_artifact = load_tuning_artifact(
        tuning_artifact_path, manifest, read_split_secret(split_secret_path)
    )
    train_records = records_for_tuning(manifest, ["train"], tuning_artifact)
    validation_records = records_for_tuning(manifest, ["validation"], tuning_artifact)
    # The external artifact physically contains only train/validation rows. No
    # source dataset object (and therefore no sealed row) is loaded here.
    train_dataset = Dataset.from_list(train_records)
    validation_dataset = Dataset.from_list(validation_records)

    encoder = matrix["encoders"][candidate["encoder"]]
    tokenizer = AutoTokenizer.from_pretrained(encoder["repository"], revision=encoder["revision"])
    model = AutoModelForSequenceClassification.from_pretrained(
        encoder["repository"],
        revision=encoder["revision"],
        num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id={label: index for index, label in enumerate(LABELS)},
        ignore_mismatched_sizes=True,
    )
    model = _apply_adaptation(
        model,
        candidate["adaptation"],
        matrix["adaptation"][candidate["adaptation"]],
    )

    max_length = int(matrix["defaults"]["max_length"])

    def prepare(batch: dict[str, list[Any]]) -> dict[str, Any]:
        encoded = tokenizer(batch["tweet"], truncation=True, max_length=max_length)
        encoded["labels"] = [int(value) for value in batch["class"]]
        encoded["soft_labels"] = [
            [
                float(hate) / float(total),
                float(offensive) / float(total),
                float(neither) / float(total),
            ]
            for hate, offensive, neither, total in zip(
                batch["hate_speech_count"],
                batch["offensive_language_count"],
                batch["neither_count"],
                batch["count"],
            )
        ]
        return encoded

    remove_columns = list(train_dataset.column_names)
    train_tokenized = train_dataset.map(prepare, batched=True, remove_columns=remove_columns)
    validation_tokenized = validation_dataset.map(prepare, batched=True, remove_columns=remove_columns)
    weights = torch.tensor(
        _class_weights(
            [int(value) for value in train_dataset["class"]],
            candidate["loss"],
            float(matrix["loss"].get("class_balanced", {}).get("beta", 0.9999)),
        )
    )
    loss_name = candidate["loss"]
    focal_gamma = float(matrix["loss"].get("focal", {}).get("gamma", 2.0))

    class ProtocolTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            soft_labels = inputs.pop("soft_labels")
            outputs = model(**inputs)
            logits = outputs.logits
            if loss_name == "soft_vote":
                targets = soft_labels.to(logits.device, dtype=logits.dtype)
                loss = -(targets * functional.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            elif loss_name == "focal":
                unweighted = functional.cross_entropy(logits, labels, reduction="none")
                probability = torch.exp(-unweighted)
                example_weights = weights.to(logits.device)[labels]
                loss = (((1 - probability) ** focal_gamma) * unweighted * example_weights).mean()
            else:
                loss = functional.cross_entropy(logits, labels, weight=weights.to(logits.device))
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(prediction) -> dict[str, float]:
        predictions = np.asarray(prediction.predictions).argmax(axis=-1)
        values = classification_metrics(prediction.label_ids, predictions)
        return {
            "accuracy": values["accuracy"],
            "macro_f1": values["macro_f1"],
            "hate_speech_precision": values["hate_speech_precision"],
            "hate_speech_recall": values["hate_speech_recall"],
            "hate_speech_f1": values["hate_speech_f1"],
        }

    arguments: dict[str, Any] = {
        "output_dir": str(output_dir),
        "seed": seed,
        "data_seed": seed,
        "learning_rate": float(matrix["adaptation"][candidate["adaptation"]]["learning_rate"]),
        "per_device_train_batch_size": int(matrix["defaults"]["train_batch_size"]),
        "per_device_eval_batch_size": int(matrix["defaults"]["eval_batch_size"]),
        "gradient_accumulation_steps": int(matrix["defaults"]["gradient_accumulation_steps"]),
        "num_train_epochs": float(matrix["defaults"]["epochs"]),
        "weight_decay": float(matrix["defaults"]["weight_decay"]),
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "report_to": [],
        "save_total_limit": 1,
        "remove_unused_columns": False,
        "fp16": matrix["defaults"]["mixed_precision"] == "fp16" and torch.cuda.is_available(),
    }
    training_argument_parameters = inspect.signature(TrainingArguments).parameters
    warmup_ratio = float(matrix["defaults"]["warmup_ratio"])
    if "warmup_ratio" in training_argument_parameters:
        arguments["warmup_ratio"] = warmup_ratio
    elif "warmup_steps" in training_argument_parameters:
        # Transformers 5 accepts a float in [0, 1) here as a total-step ratio.
        arguments["warmup_steps"] = warmup_ratio
    else:
        raise RuntimeError("pinned Transformers exposes no warmup configuration")
    if "save_only_model" not in training_argument_parameters:
        raise RuntimeError("pinned Transformers must support save_only_model for bounded disk use")
    arguments["save_only_model"] = True
    evaluation_key = "eval_strategy" if "eval_strategy" in training_argument_parameters else "evaluation_strategy"
    arguments[evaluation_key] = "epoch"
    trainer_arguments: dict[str, Any] = {
        "model": model,
        "args": TrainingArguments(**arguments),
        "train_dataset": train_tokenized,
        "eval_dataset": validation_tokenized,
        "data_collator": DataCollatorWithPadding(tokenizer),
        "compute_metrics": compute_metrics,
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_arguments["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        trainer_arguments["tokenizer"] = tokenizer
    else:
        raise RuntimeError("pinned Transformers Trainer exposes no tokenizer/processor binding")
    trainer = ProtocolTrainer(**trainer_arguments)
    trainer.train()
    prediction_output = trainer.predict(validation_tokenized)
    checkpoint_dir = output_dir / "selected_checkpoint"
    if candidate["adaptation"] == "lora":
        # Export consumes a standalone Transformers checkpoint, not a PEFT adapter.
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(checkpoint_dir, safe_serialization=True)
        checkpoint_format = "merged_full_model"
    else:
        trainer.save_model(str(checkpoint_dir))
        checkpoint_format = "full_model"
    tokenizer.save_pretrained(str(checkpoint_dir))
    # Keep only the standalone export checkpoint; Trainer's resumable epoch
    # checkpoint can otherwise duplicate several gigabytes per matrix run.
    for trainer_checkpoint in output_dir.glob("checkpoint-*"):
        if trainer_checkpoint.is_dir() and trainer_checkpoint.parent == output_dir:
            shutil.rmtree(trainer_checkpoint)
    validation_predictions = np.asarray(prediction_output.predictions).argmax(axis=-1).tolist()
    validation_labels = [int(value) for value in validation_dataset["class"]]
    validation_groups = [str(record["group_token"]) for record in validation_records]
    metrics = classification_metrics(validation_labels, validation_predictions)
    bootstrap_resamples = int(matrix["selection"]["required_bootstrap_resamples"])
    slice_configuration = json.loads(Path(slices_path).read_text(encoding="utf-8"))
    checkpoint_inventory = artifact_tree(checkpoint_dir)
    result: dict[str, Any] = {
        "schema_version": 2,
        "protocol_id": "grouped_v1",
        "run_id": f"{candidate_id}__seed_{seed}",
        "candidate_id": candidate_id,
        "seed": seed,
        "evaluated_split": "validation",
        "frozen_test_loaded": False,
        "public_manifest_payload_sha256": manifest["integrity"]["payload_sha256"],
        "matrix_sha256": file_sha256(matrix_path),
        "matrix_semantic_sha256": semantic_file_sha256(matrix_path),
        "slices_sha256": file_sha256(slices_path),
        "slices_semantic_sha256": semantic_file_sha256(slices_path),
        "candidate": candidate,
        "encoder": encoder,
        "training_hyperparameters": {
            **matrix["defaults"],
            "learning_rate": float(matrix["adaptation"][candidate["adaptation"]]["learning_rate"]),
        },
        "metrics": metrics,
        "grouped_bootstrap": {
            "macro_f1": grouped_bootstrap_interval(
                validation_labels,
                validation_predictions,
                validation_groups,
                macro_f1_value,
                seed=seed,
                resamples=bootstrap_resamples,
            ),
            "hate_speech_recall": grouped_bootstrap_interval(
                validation_labels,
                validation_predictions,
                validation_groups,
                hate_recall_value,
                seed=seed,
                resamples=bootstrap_resamples,
            ),
        },
        "slice_metrics": slice_metric_report(
            [dict(row) for row in validation_dataset],
            validation_labels,
            validation_predictions,
            slice_configuration,
        ),
        "checkpoint": "selected_checkpoint",
        "checkpoint_format": checkpoint_format,
        "checkpoint_artifacts": checkpoint_inventory,
        "checkpoint_storage": {
            "save_only_model": True,
            "resumable_optimizer_state": False,
            "transient_best_model_checkpoint_removed": True,
            "matrix_runner_retention": (
                "non-production checkpoints are pruned; only the current complete-candidate "
                "leader may retain seed-42 standalone weights"
            ),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "peft": importlib.metadata.version("peft"),
            "datasets": importlib.metadata.version("datasets"),
        },
    }
    result["integrity"] = {"payload_sha256": payload_sha256(result)}
    (output_dir / "run_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
