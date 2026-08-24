import json
from pathlib import Path

import pytest

from hate_speech_adv.frozen_state import verify_repository_frozen_state
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "evaluation/manifests/grouped_v1.json").read_text(encoding="utf-8"))


def _configuration(root: Path) -> Path:
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(MANIFEST), encoding="utf-8")
    config = {
        "schema_version": 2,
        "protocol_id": "grouped_v1",
        "manifest_path": "manifest.json",
        "result_path": "result.json",
        "reveal_path": "reveal.json",
        "selection_receipt_path": "selection.json",
        "export_manifest_path": "export.json",
        "quality_gate_path": "gate.json",
        "export_config_path": "export-config.json",
        "slices_path": "slices.json",
        "allowed_states": [
            "pre_test_sealed",
            "post_test_unrevealed",
            "post_test_revealed",
        ],
    }
    path = root / "state.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_frozen_state_verifier_accepts_pre_test_state(tmp_path):
    configuration = _configuration(tmp_path)
    assert verify_repository_frozen_state(configuration, tmp_path)["state"] == "pre_test_sealed"


def test_frozen_state_verifier_rejects_reveal_before_result(tmp_path):
    configuration = _configuration(tmp_path)
    (tmp_path / "reveal.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot exist before"):
        verify_repository_frozen_state(configuration, tmp_path)
