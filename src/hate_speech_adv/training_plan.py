from __future__ import annotations

import contextlib
import json
import importlib.metadata
import os
import socket
import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from .integrity import file_sha256, payload_sha256, semantic_file_sha256
from .manifest import load_manifest

ALLOWED_ADAPTATIONS = frozenset(("lora", "partial", "full"))
ALLOWED_LOSSES = frozenset(("weighted_ce", "focal", "class_balanced", "soft_vote"))
REQUIRED_SEEDS = (13, 42, 73)
SECURE_TRAINING_STACK = {
    "torch": "2.13.0",
    "transformers": "5.15.1",
    "peft": "0.20.0",
    "accelerate": "1.14.0",
}


def load_matrix(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_matrix(matrix: dict[str, Any]) -> None:
    if matrix.get("protocol_id") != "grouped_v1":
        raise ValueError("training matrix must target grouped_v1")
    if tuple(matrix.get("seeds", ())) != REQUIRED_SEEDS:
        raise ValueError(f"training seeds must be exactly {REQUIRED_SEEDS}")
    selection = matrix.get("selection", {})
    if int(selection.get("production_seed", -1)) != 42:
        raise ValueError("the declared production seed must be 42")
    if int(selection.get("required_bootstrap_resamples", 0)) < 1000:
        raise ValueError("grouped bootstrap must use at least 1000 resamples")
    encoders = matrix.get("encoders", {})
    candidates = matrix.get("candidates", [])
    if not candidates:
        raise ValueError("training matrix has no candidates")
    candidate_ids: set[str] = set()
    coverage = {"encoders": set(), "adaptations": set(), "losses": set()}
    for candidate in candidates:
        candidate_id = str(candidate["id"])
        if candidate_id in candidate_ids:
            raise ValueError(f"duplicate candidate ID: {candidate_id}")
        candidate_ids.add(candidate_id)
        if candidate["encoder"] not in encoders:
            raise ValueError(f"unknown encoder in {candidate_id}")
        if candidate["adaptation"] not in ALLOWED_ADAPTATIONS:
            raise ValueError(f"unknown adaptation in {candidate_id}")
        if candidate["loss"] not in ALLOWED_LOSSES:
            raise ValueError(f"unknown loss in {candidate_id}")
        coverage["encoders"].add(candidate["encoder"])
        coverage["adaptations"].add(candidate["adaptation"])
        coverage["losses"].add(candidate["loss"])
    if coverage["encoders"] != set(encoders):
        raise ValueError("not every configured encoder is represented")
    if coverage["adaptations"] != set(ALLOWED_ADAPTATIONS):
        raise ValueError("matrix must cover LoRA, partial, and full fine-tuning")
    if coverage["losses"] != set(ALLOWED_LOSSES):
        raise ValueError("matrix must cover all four loss strategies")
    if "learning_rate" in matrix.get("defaults", {}):
        raise ValueError("learning rate must be adaptation-specific")
    for adaptation in ALLOWED_ADAPTATIONS:
        learning_rate = float(matrix.get("adaptation", {}).get(adaptation, {}).get("learning_rate", 0))
        if not 0 < learning_rate < 0.01:
            raise ValueError(f"adaptation {adaptation} needs a valid learning rate")
    for name, encoder in encoders.items():
        if not encoder.get("repository") or len(str(encoder.get("revision", ""))) != 40:
            raise ValueError(f"encoder {name} must be pinned to a 40-character Hub revision")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def smoke_loss_values() -> dict[str, float]:
    """Exercise every loss contract without importing torch or downloading a model."""
    logits = np.asarray([[2.0, 0.5, -1.0], [0.1, 1.2, 0.2], [-0.2, 0.4, 1.4]])
    labels = np.asarray([0, 1, 2])
    votes = np.asarray([[2, 1, 0], [0, 3, 0], [0, 1, 2]], dtype=np.float64)
    probabilities = np.clip(_softmax(logits), 1e-12, 1.0)
    weights = np.asarray([2.0, 0.5, 1.0])
    selected = probabilities[np.arange(len(labels)), labels]
    weighted_ce = float((-np.log(selected) * weights[labels]).mean())
    focal = float((-(1 - selected) ** 2 * np.log(selected) * weights[labels]).mean())
    counts = np.asarray([100.0, 1000.0, 300.0])
    beta = 0.9999
    class_balanced_weights = (1 - beta) / (1 - np.power(beta, counts))
    class_balanced_weights /= class_balanced_weights.mean()
    class_balanced = float((-np.log(selected) * class_balanced_weights[labels]).mean())
    soft_targets = votes / votes.sum(axis=1, keepdims=True)
    soft_vote = float((-(soft_targets * np.log(probabilities)).sum(axis=1)).mean())
    values = {
        "weighted_ce": weighted_ce,
        "focal": focal,
        "class_balanced": class_balanced,
        "soft_vote": soft_vote,
    }
    if not all(np.isfinite(value) and value > 0 for value in values.values()):
        raise RuntimeError("loss smoke contract produced an invalid value")
    return values


def tiny_offline_training_step(seed: int = 13) -> dict[str, Any]:
    """Take one real ``Trainer`` + PEFT step using an in-memory local model."""
    installed = {
        package: importlib.metadata.version(package) for package in SECURE_TRAINING_STACK
    }
    if installed != SECURE_TRAINING_STACK:
        raise RuntimeError(
            f"secure training stack mismatch: expected {SECURE_TRAINING_STACK}, found {installed}"
        )

    import accelerate  # noqa: F401 - import proves runtime compatibility
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        RobertaConfig,
        RobertaForSequenceClassification,
        Trainer,
        TrainingArguments,
    )

    torch.manual_seed(seed)
    configuration = RobertaConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout=0.0,
        num_labels=3,
    )
    base_model = RobertaForSequenceClassification(configuration)
    model = get_peft_model(
        base_model,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["query", "value"],
            modules_to_save=["classifier"],
        ),
    )
    input_ids = torch.tensor(
        [
            [0, 5, 9, 2, 1, 1, 1, 1],
            [0, 7, 8, 3, 2, 1, 1, 1],
            [0, 4, 6, 10, 11, 2, 1, 1],
            [0, 12, 13, 14, 2, 1, 1, 1],
            [0, 15, 16, 2, 1, 1, 1, 1],
            [0, 17, 18, 19, 20, 2, 1, 1],
        ],
        dtype=torch.long,
    )
    attention_mask = input_ids.ne(1).long()
    # A single-class micro-batch gives the one-step smoke a clear descent
    # direction; coverage of all three labels/losses is exercised separately.
    labels = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    before_parameters = [parameter.detach().clone() for parameter in trainable]
    model.eval()
    with torch.no_grad():
        before_tensor = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        ).loss

    class TinyDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(labels)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return {
                "input_ids": input_ids[index],
                "attention_mask": attention_mask[index],
                "labels": labels[index],
            }

    previous_environment = {
        name: os.environ.get(name)
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "WANDB_DISABLED")
    }
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_DISABLED": "true",
        }
    )
    try:
        with tempfile.TemporaryDirectory(prefix="grouped-v1-trainer-smoke-") as output_dir:
            arguments = TrainingArguments(
                output_dir=output_dir,
                max_steps=1,
                per_device_train_batch_size=len(labels),
                learning_rate=0.05,
                weight_decay=0.0,
                warmup_steps=0,
                logging_strategy="no",
                save_strategy="no",
                eval_strategy="no",
                report_to="none",
                disable_tqdm=True,
                use_cpu=True,
                seed=seed,
                data_seed=seed,
                dataloader_num_workers=0,
                remove_unused_columns=False,
            )
            trainer = Trainer(model=model, args=arguments, train_dataset=TinyDataset())

            def deny_network(*_args, **_kwargs):
                raise RuntimeError("network access is forbidden in the offline training smoke")

            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(socket.socket, "connect", deny_network))
                stack.enter_context(mock.patch.object(socket.socket, "connect_ex", deny_network))
                stack.enter_context(mock.patch("socket.create_connection", deny_network))
                train_output = trainer.train()
    finally:
        for name, previous in previous_environment.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous

    model.eval()
    with torch.no_grad():
        after_tensor = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        ).loss
    before = float(before_tensor.detach())
    after = float(after_tensor.detach())
    parameters_changed = any(
        not torch.equal(before_parameter, after_parameter.detach())
        for before_parameter, after_parameter in zip(before_parameters, trainable)
    )
    finite_parameters = all(bool(torch.isfinite(parameter).all()) for parameter in trainable)
    if (
        int(trainer.state.global_step) != 1
        or not parameters_changed
        or not finite_parameters
        or not np.isfinite(after)
        or after >= before
    ):
        raise RuntimeError(
            "tiny offline Trainer + PEFT step failed its update/loss gate: "
            f"step={trainer.state.global_step}, changed={parameters_changed}, "
            f"finite={finite_parameters}, before={before}, after={after}"
        )
    return {
        "model": "tiny_local_roberta_lora_v1",
        "training_api": "transformers.Trainer",
        "adaptation_api": "peft.get_peft_model",
        "network_accessed": False,
        "network_enforcement": "offline environment plus socket denial during Trainer.train",
        "secure_stack": installed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "examples": len(labels),
        "steps": int(trainer.state.global_step),
        "trainer_reported_loss": float(train_output.training_loss),
        "parameters_changed": parameters_changed,
        "finite_parameters": finite_parameters,
        "loss_before": before,
        "loss_after": after,
    }


def build_run_plan(matrix_path: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    matrix_path = Path(matrix_path)
    matrix = load_matrix(matrix_path)
    validate_matrix(matrix)
    manifest = load_manifest(manifest_path)
    support = manifest["summary"]["split_support"]
    train_count = int(support["train"]["rows"])
    validation_count = int(support["validation"]["rows"])
    runs = [
        {
            "run_id": f"{candidate['id']}__seed_{seed}",
            "candidate_id": candidate["id"],
            "seed": seed,
            "encoder": candidate["encoder"],
            "adaptation": candidate["adaptation"],
            "loss": candidate["loss"],
            "learning_rate": float(matrix["adaptation"][candidate["adaptation"]]["learning_rate"]),
            "allowed_splits": ["train", "validation"],
        }
        for candidate in matrix["candidates"]
        for seed in matrix["seeds"]
    ]
    plan: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": "grouped_v1",
        "public_manifest_payload_sha256": manifest["integrity"]["payload_sha256"],
        "matrix_sha256": file_sha256(matrix_path),
        "matrix_semantic_sha256": semantic_file_sha256(matrix_path),
        "frozen_test_loaded": False,
        "tuning_rows": {"train": train_count, "validation": validation_count},
        "loss_smoke": smoke_loss_values(),
        "offline_training_smoke": tiny_offline_training_step(),
        "runs": runs,
    }
    plan["integrity"] = {"payload_sha256": payload_sha256(plan)}
    return plan
