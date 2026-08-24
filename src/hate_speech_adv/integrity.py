from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def payload_sha256(document: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {key: value for key, value in document.items() if key != "integrity"}
    )


def verify_payload_integrity(document: Mapping[str, Any]) -> None:
    expected = str(document.get("integrity", {}).get("payload_sha256", ""))
    actual = payload_sha256(document)
    if expected != actual:
        raise ValueError("document payload hash does not match")


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def semantic_file_sha256(path: str | Path) -> str:
    """Hash executable/configuration meaning while ignoring harmless formatting."""
    path = Path(path)
    if path.suffix == ".json":
        return canonical_json_sha256(json.loads(path.read_text(encoding="utf-8")))
    if path.suffix == ".py":
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        return hashlib.sha256(
            ast.dump(tree, annotate_fields=True, include_attributes=False).encode("utf-8")
        ).hexdigest()
    return file_sha256(path)


def semantic_bundle(path_pairs: Iterable[tuple[str, str | Path]]) -> dict[str, Any]:
    files = [
        {
            "path": logical_path,
            "sha256": file_sha256(path),
            "semantic_sha256": semantic_file_sha256(path),
        }
        for logical_path, path in sorted(path_pairs)
    ]
    return {"files": files, "semantic_tree_sha256": canonical_json_sha256(files)}


def artifact_tree(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = [
        {
            "path": str(item.relative_to(root)),
            "size_bytes": item.stat().st_size,
            "sha256": file_sha256(item),
        }
        for item in sorted(root.rglob("*"))
        if item.is_file()
    ]
    if not files:
        raise ValueError(f"artifact directory is empty: {root}")
    return {
        "files": files,
        "tree_sha256": canonical_json_sha256(files),
        "size_bytes": sum(int(item["size_bytes"]) for item in files),
    }


def artifact_descriptor(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if target.is_file():
        return {
            "kind": "file",
            "sha256": file_sha256(target),
            "size_bytes": target.stat().st_size,
        }
    tree = artifact_tree(target)
    return {"kind": "directory", **tree}
