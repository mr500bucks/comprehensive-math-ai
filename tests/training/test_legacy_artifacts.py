from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPOSITORY_ROOT / "training" / "legacy_artifact_status.json"


def test_legacy_training_inventory_fingerprints_every_preserved_file() -> None:
    manifest = cast(dict[str, Any], json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))
    assert manifest["status"] == "legacy_partially_unreproducible"
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
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        assert digest == record["sha256"], f"historical artifact drifted: {record['path']}"
