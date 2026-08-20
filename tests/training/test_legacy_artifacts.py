from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPOSITORY_ROOT / "training" / "legacy_artifact_status.json"


def _canonical_text_sha256(path: Path) -> str:
    """Hash UTF-8 historical text with platform-independent LF newlines."""
    text = path.read_text(encoding="utf-8")
    canonical_bytes = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def test_canonical_text_fingerprint_is_independent_of_checkout_newlines(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_bytes(b"first\nsecond\n")
    lf_digest = _canonical_text_sha256(artifact)

    artifact.write_bytes(b"first\r\nsecond\r\n")

    assert _canonical_text_sha256(artifact) == lf_digest


def test_legacy_training_inventory_fingerprints_every_preserved_file() -> None:
    manifest = cast(dict[str, Any], json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))
    assert manifest["status"] == "legacy_partially_unreproducible"
    assert manifest["fingerprint_algorithm"] == "sha256"
    assert manifest["fingerprint_normalization"] == "utf-8-lf"
    records = cast(list[dict[str, str]], manifest["available_files"])

    recorded_paths = {record["path"] for record in records}
    actual_paths = {
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for dated_directory in ("20260720", "20260724")
        for path in (REPOSITORY_ROOT / "training" / dated_directory).rglob("*")
        if path.is_file()
    }
    assert recorded_paths == actual_paths

    for record in records:
        artifact = REPOSITORY_ROOT / record["path"]
        digest = _canonical_text_sha256(artifact)
        assert digest == record["sha256"], f"historical artifact drifted: {record['path']}"
