from pathlib import Path
import hashlib
import hmac
import json

import pytest

from hate_speech_adv.manifest import (
    FROZEN_TEST_ACKNOWLEDGEMENT,
    authorize_frozen_test,
    create_one_shot_ledger,
    load_manifest,
    load_tuning_artifact,
    records_for_tuning,
    require_external_private_path,
)
from hate_speech_adv.integrity import canonical_json_bytes, payload_sha256

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "evaluation/manifests/grouped_v1.json"


def test_tuning_can_load_train_and_validation_only(tmp_path):
    manifest = load_manifest(MANIFEST_PATH)
    secret = b"s" * 32
    private = {
        "schema_version": 1,
        "protocol_id": "grouped_v1",
        "scope": ["train", "validation"],
        "contains_frozen_test_membership": False,
        "columns": [
            "source_row_id",
            "row_token",
            "split_code",
            "group_token",
            "count",
            "hate_speech_count",
            "offensive_language_count",
            "neither_count",
            "class",
            "tweet",
        ],
        "rows": [
            [0, "a" * 64, manifest["protocol"]["split_codes"]["train"], "c" * 64, 3, 1, 1, 1, 0, "train"],
            [1, "b" * 64, manifest["protocol"]["split_codes"]["validation"], "d" * 64, 3, 0, 2, 1, 1, "validation"],
        ],
    }
    commitment = hmac.new(
        secret,
        b"grouped_v1\0tuning-membership-v1\0" + canonical_json_bytes(private),
        hashlib.sha256,
    ).hexdigest()
    private["membership_commitment"] = commitment
    private["integrity"] = {"payload_sha256": payload_sha256(private)}
    private_path = tmp_path / "private.json"
    private_path.write_text(json.dumps(private), encoding="utf-8")
    manifest["sealed_tuning_membership"]["rows"] = 2
    manifest["sealed_tuning_membership"]["commitment"] = commitment
    private = load_tuning_artifact(private_path, manifest, secret)
    assert records_for_tuning(manifest, ["train"], private)
    assert records_for_tuning(manifest, ["validation"], private)
    with pytest.raises(PermissionError, match="private train/validation"):
        records_for_tuning(manifest, ["train"])
    with pytest.raises(PermissionError, match="frozen test"):
        records_for_tuning(manifest, ["test"], private)
    with pytest.raises(PermissionError, match="frozen test"):
        records_for_tuning(manifest, ["train", "test"], private)
    assert all(row[2] != manifest["protocol"]["split_codes"]["test"] for row in private["rows"])
    tampered = json.loads(private_path.read_text(encoding="utf-8"))
    tampered["rows"][0][-1] = "changed"
    tampered["integrity"] = {"payload_sha256": payload_sha256(tampered)}
    private_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(PermissionError, match="does not open"):
        load_tuning_artifact(private_path, manifest, secret)


def test_private_tuning_artifact_must_be_external():
    with pytest.raises(ValueError, match="outside the repository"):
        require_external_private_path(
            ROOT / "evaluation/private/grouped_v1_tv.json",
            ROOT,
            "private train/validation artifact",
        )


def test_frozen_loader_rejects_missing_acknowledgement(tmp_path):
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    with pytest.raises(PermissionError, match="acknowledgement"):
        authorize_frozen_test(
            MANIFEST_PATH,
            receipt,
            ROOT,
            configuration_path=tmp_path / "config.json",
            split_secret_path=tmp_path / "secret",
            export_manifest_path=tmp_path / "export.json",
            quality_gate_path=tmp_path / "gate.json",
            export_root=tmp_path / "export",
            dataset_loader=lambda *_: None,
            ledger_path=tmp_path / "ledger.json",
            acknowledgement=None,
        )
    assert not (tmp_path / "ledger.json").exists()


def test_uncommitted_receipt_cannot_open_test(tmp_path):
    receipt = tmp_path / "receipt.json"
    from hate_speech_adv.integrity import payload_sha256

    document = {
        "schema_version": 2,
        "protocol_id": "grouped_v1",
        "public_manifest_payload_sha256": load_manifest(MANIFEST_PATH)["integrity"]["payload_sha256"],
        "evaluated_split": "validation",
        "selection_locked": False,
        "frozen_test_evaluated": False,
    }
    document["integrity"] = {"payload_sha256": payload_sha256(document)}
    receipt.write_text(__import__("json").dumps(document), encoding="utf-8")
    with pytest.raises(PermissionError, match="selection must be locked"):
        authorize_frozen_test(
            MANIFEST_PATH,
            receipt,
            ROOT,
            configuration_path=tmp_path / "config.json",
            split_secret_path=tmp_path / "secret",
            export_manifest_path=tmp_path / "export.json",
            quality_gate_path=tmp_path / "gate.json",
            export_root=tmp_path / "export",
            dataset_loader=lambda *_: None,
            ledger_path=tmp_path / "ledger.json",
            acknowledgement=FROZEN_TEST_ACKNOWLEDGEMENT,
        )
    assert (tmp_path / "ledger.json").exists()


def test_one_shot_ledger_is_atomic_and_non_reusable(tmp_path):
    ledger = tmp_path / "opening.json"
    create_one_shot_ledger(ledger, {"status": "consumed"})
    assert ledger.stat().st_mode & 0o077 == 0
    with pytest.raises(FileExistsError):
        create_one_shot_ledger(ledger, {"status": "second"})


def test_opening_ledger_is_durable_before_dataset_loader(monkeypatch, tmp_path):
    import hate_speech_adv.manifest as manifest_module
    import hate_speech_adv.artifact_evaluation as artifact_module
    import hate_speech_adv.export_pipeline as export_module
    import hate_speech_adv.protocol as protocol_module

    manifest = load_manifest(MANIFEST_PATH)
    receipt_path = tmp_path / "receipt.json"
    receipt = {
        "public_manifest_payload_sha256": manifest["integrity"]["payload_sha256"],
        "evaluated_split": "validation",
        "selection_locked": True,
        "frozen_test_evaluated": False,
        "protocol_lock": {"commit": "a" * 40},
        "result_lock": {"commit": "b" * 40},
    }
    receipt["integrity"] = {"payload_sha256": payload_sha256(receipt)}
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"dataset": {"repository": "fixture", "revision": "revision"}}),
        encoding="utf-8",
    )
    secret_path = tmp_path / "secret"
    secret_path.write_text("ab" * 32 + "\n", encoding="ascii")
    secret_path.chmod(0o600)
    ledger_path = tmp_path / "ledger.json"
    export_manifest_path = tmp_path / "export.json"
    export_manifest_path.write_text("{}", encoding="utf-8")
    quality_gate_path = tmp_path / "gate.json"
    quality_gate_path.write_text("{}", encoding="utf-8")
    events = []

    monkeypatch.setattr(
        manifest_module,
        "verify_locked_selection_receipt",
        lambda *_: (receipt, "f" * 40),
    )
    monkeypatch.setattr(manifest_module, "verify_file_is_committed_at_head", lambda *_: "f" * 40)
    monkeypatch.setattr(export_module, "validate_export_manifest_and_tree", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(artifact_module, "validate_quality_gate_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        protocol_module,
        "reconstruct_and_verify_sealed_partition",
        lambda *_: [],
    )

    def dataset_loader(*_):
        assert ledger_path.exists()
        events.append("dataset_loaded_after_ledger")
        return object()

    authorization = authorize_frozen_test(
        MANIFEST_PATH,
        receipt_path,
        ROOT,
        configuration_path=config_path,
        split_secret_path=secret_path,
        export_manifest_path=export_manifest_path,
        quality_gate_path=quality_gate_path,
        export_root=tmp_path / "export",
        dataset_loader=dataset_loader,
        ledger_path=ledger_path,
        acknowledgement=FROZEN_TEST_ACKNOWLEDGEMENT,
    )
    assert authorization["test_records"] == []
    assert events == ["dataset_loaded_after_ledger"]


def test_opening_consumes_ledger_before_even_reading_manifest(tmp_path):
    ledger_path = tmp_path / "ledger.json"
    with pytest.raises(FileNotFoundError):
        authorize_frozen_test(
            tmp_path / "missing-manifest.json",
            tmp_path / "missing-receipt.json",
            ROOT,
            configuration_path=tmp_path / "missing-config.json",
            split_secret_path=tmp_path / "missing-secret",
            export_manifest_path=tmp_path / "missing-export.json",
            quality_gate_path=tmp_path / "missing-gate.json",
            export_root=tmp_path / "missing-export",
            dataset_loader=lambda *_: pytest.fail("dataset loader must not run"),
            ledger_path=ledger_path,
            acknowledgement=FROZEN_TEST_ACKNOWLEDGEMENT,
        )
    assert ledger_path.exists()
    assert json.loads(ledger_path.read_text(encoding="utf-8"))["status"] == (
        "consumed_before_any_dataset_or_partition_access"
    )
