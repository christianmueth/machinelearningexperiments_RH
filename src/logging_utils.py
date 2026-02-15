import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunMetadata:
    started_utc: str
    hostname: str
    platform: str
    python: str
    git_commit: str | None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def get_git_commit(workspace_root: str) -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(workspace_root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        s = (r.stdout or "").strip()
        return s if s else None
    except Exception:  # noqa: BLE001
        return None


def make_run_dir(runs_root: str, *, tag: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(tag))
    out = Path(runs_root) / f"{ts}__{safe}"
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


def write_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


def snapshot_run_metadata(workspace_root: str) -> RunMetadata:
    import sys

    return RunMetadata(
        started_utc=_utc_now_iso(),
        hostname=platform.node(),
        platform=platform.platform(),
        python=sys.version.replace("\n", " "),
        git_commit=get_git_commit(workspace_root),
    )


def save_run_snapshot(out_dir: str, *, config: dict[str, Any], workspace_root: str) -> None:
    meta = snapshot_run_metadata(workspace_root)
    write_json(os.path.join(out_dir, "run_meta.json"), asdict(meta))
    write_json(os.path.join(out_dir, "config.json"), config)
