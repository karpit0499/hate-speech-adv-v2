import json
from pathlib import Path
import subprocess

from hate_speech_adv.manifest import load_manifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "evaluation/manifests/grouped_v1.json"
AUDIT_PATH = ROOT / "evaluation/audits/grouped_v1_audit.json"


def test_committed_manifest_integrity_and_no_text():
    manifest = load_manifest(MANIFEST_PATH)
    assert manifest["summary"]["total_source_rows"] == 24_783
    assert manifest["summary"]["tuning_rows"] == 22_305
    assert manifest["summary"]["total_groups"] == 24_556
    assert "rows" not in manifest
    assert "columns" not in manifest
    assert manifest["sealed_tuning_membership"]["membership_published"] is False
    assert manifest["sealed_test"]["membership_published"] is False
    assert len(MANIFEST_PATH.read_bytes()) < 100_000


def test_partition_is_complete_and_approximately_stratified():
    manifest = load_manifest(MANIFEST_PATH)
    supports = manifest["summary"]["split_support"]
    assert sum(value["rows"] for value in supports.values()) == 24_783
    for split, expected_ratio in (("train", 0.8), ("validation", 0.1)):
        assert abs(supports[split]["rows"] / manifest["summary"]["total_source_rows"] - expected_ratio) < 0.005
    assert manifest["sealed_test"]["rows"] == supports["test"]["rows"]


def test_audit_records_the_known_duplicates_and_ambiguity_without_leakage():
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    strict = audit["duplicates"]["strict_v1"]
    template = audit["duplicates"]["template_v1"]
    assert strict["duplicate_groups"] == 12
    assert strict["conflicting_label_groups"] == 1
    assert template["duplicate_groups"] == 121
    assert template["rows_in_duplicate_groups"] == 348
    assert template["conflicting_label_groups"] == 12
    assert template["rows_in_conflicting_groups"] == 51
    assert template["cross_partition_groups"] == 0
    assert audit["ambiguity"]["all"]["non_unanimous"] == 7_301
    assert audit["ambiguity"]["all"]["by_class"]["hate_speech"]["non_unanimous"] == 1_167
    low_consensus = audit["slice_counts"]["ambiguity_slices"]["low_consensus"]
    assert sum(low_consensus.values()) == 71
    assert audit["contains_row_membership"] is False


def test_no_versioned_json_discloses_exact_grouped_v1_membership():
    paths = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "*.json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    def inspect(value, path):
        if isinstance(value, dict):
            assert "source_row_id" not in value, path
            assert "split_code" not in value, path
            forbidden_membership_keys = {
                "source_row_ids",
                "row_ids",
                "train_ids",
                "validation_ids",
                "test_ids",
                "split_assignments",
                "split_map",
                "partition_map",
            }
            assert forbidden_membership_keys.isdisjoint(value), path
            columns = value.get("columns")
            if isinstance(columns, list):
                assert "source_row_id" not in columns, path
                assert "split_code" not in columns, path
            for nested in value.values():
                inspect(nested, path)
        elif isinstance(value, list):
            for nested in value:
                inspect(nested, path)

    for relative in paths:
        path = ROOT / relative
        if path.exists():
            inspect(json.loads(path.read_text(encoding="utf-8")), relative)

    split_config = json.loads((ROOT / "configs/data/grouped_v1.json").read_text(encoding="utf-8"))
    assert "seed" not in split_config["partition"]
    assert "random_state" not in split_config["partition"]


def test_sensitive_membership_and_ledger_filename_variants_are_ignored():
    candidates = (
        "scratch/grouped_v1_tv_backup.json",
        "scratch/grouped_v1_train_membership.json",
        "scratch/grouped_v1_opened_test.json",
        "scratch/grouped_v1-opening-ledger.json",
        "scratch/grouped_v1_split_secret.txt",
        "scratch/grouped-v1-split-secret.txt",
        "scratch/frozen-test-membership.parquet",
        "scratch/train-validation-split-map.csv",
        "scratch/alternate_partition_map.jsonl",
        "scratch/frozen_test_opened_test.arrow",
        "scratch/test-one-shot-ledger.txt",
    )
    for candidate in candidates:
        completed = subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", candidate],
            cwd=ROOT,
        )
        assert completed.returncode == 0, candidate
