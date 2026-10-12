"""Download the air-raid alert datasets and record a reproducible snapshot.

    python -m src.fetch_data                    # latest version of the dataset
    python -m src.fetch_data --commit <sha>     # exactly the version recorded in a snapshot.json

Source: https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset (MIT).
The dataset repository is updated daily, so "latest" changes every day. To make
results reproducible we resolve "main" to a concrete commit hash first, download
the files AT THAT COMMIT, and store the hash in data/raw/snapshot.json. Anyone can
later re-download the identical files with --commit <hash>. The sha256 of each file
is stored as well, to verify that the files on disk are the ones that were used.

Primary file is the volunteer (eTryvoga) dataset: oblast level, updated daily.
The official dataset is downloaded too; as of 2026-10-12 it stops on 2026-09-07
and has a 4-day hole (2026-08-29 .. 2026-09-02), see features.coverage_end.
"""
import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

import requests

REPO = "Vadimkin/ukrainian-air-raid-sirens-dataset"
FILES = ["volunteer_data_en.csv", "official_data_en.csv"]
OUT = Path(__file__).resolve().parents[1] / "data" / "raw"


def resolve_commit(ref: str = "main") -> str | None:
    """Commit hash that branch `ref` points to right now; None if it cannot be resolved.

    Tries the GitHub API first, then `git ls-remote` (works where the API is blocked
    or rate-limited, as long as git is installed).
    """
    try:
        r = requests.get(f"https://api.github.com/repos/{REPO}/commits/{ref}",
                         headers={"Accept": "application/vnd.github+json"}, timeout=30)
        r.raise_for_status()
        return r.json()["sha"]
    except Exception as e:  # rate limit, blocked, no network, ...
        print(f"note: GitHub API failed ({e.__class__.__name__}), trying git ls-remote")
    try:
        out = subprocess.run(["git", "ls-remote", f"https://github.com/{REPO}", f"refs/heads/{ref}"],
                             capture_output=True, text=True, timeout=60, check=True).stdout.split()
        return out[0] if out else None
    except Exception as e:
        print(f"warning: could not resolve commit for '{ref}': {e!r}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", help="dataset commit hash to download (default: current main)")
    args = ap.parse_args()

    commit = args.commit or resolve_commit("main")
    ref = commit or "main"
    if commit is None:
        print("warning: downloading from 'main' without a commit hash; the snapshot is NOT reproducible")
    base = f"https://raw.githubusercontent.com/{REPO}/{ref}/datasets/"

    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"repository": f"https://github.com/{REPO}",
            "commit": commit,
            "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "reproduce_with": f"python -m src.fetch_data --commit {commit}" if commit else None,
            "files": {}}
    for name in FILES:
        r = requests.get(base + name, timeout=180)
        r.raise_for_status()
        (OUT / name).write_bytes(r.content)
        meta["files"][name] = {"sha256": hashlib.sha256(r.content).hexdigest(),
                               "bytes": len(r.content)}
        print(f"saved {name}: {len(r.content) / 1e6:.1f} MB")
    (OUT / "snapshot.json").write_text(json.dumps(meta, indent=2))
    print(f"snapshot written to {OUT / 'snapshot.json'} (dataset commit: {commit})")


if __name__ == "__main__":
    main()
