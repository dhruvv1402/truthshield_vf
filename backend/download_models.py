"""Fetch the trained Truth Shield weights from the GitHub release.

The phase-1 RoBERTa checkpoint and the phase-2 fusion head are published as
release assets (they are far too large for the git tree), so both the Docker
build and a local dev setup pull them with this script.

    python download_models.py            # skips anything already present
    python download_models.py --force    # re-download
"""

import argparse
import os
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = os.getenv("MODEL_RELEASE_REPO", "Gyaanendra/AIML-PROJECT-CSET312")
TAG = os.getenv("MODEL_RELEASE_TAG", "v3")
BASE = f"https://github.com/{REPO}/releases/download/{TAG}"

MODELS_DIR = Path(os.getenv("MODELS_DIR", Path(__file__).parent / "models")).resolve()

# archive name -> a path that must exist once the archive is unpacked
ASSETS = {
    "phase1_roberta_fulltune.7z": "phase1_roberta_fulltune/best/config.json",
    "phase2_fusion_head.7z": "phase2_fusion_head/fusion_head.pt",
}


def _progress(done: int, total: int, name: str) -> None:
    if total <= 0:
        return
    pct = done * 100 // total
    bar = "#" * (pct // 4)
    sys.stdout.write(f"\r  {name}  [{bar:<25}] {pct:3d}%  {done >> 20}/{total >> 20} MiB")
    sys.stdout.flush()


def download(url: str, dest: Path) -> None:
    print(f">> downloading {dest.name}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            _progress(done, total, dest.name)
    print()
    tmp.replace(dest)


def extract(archive: Path, target: Path) -> None:
    print(f">> extracting {archive.name}")
    import py7zr

    with py7zr.SevenZipFile(archive, mode="r") as z:
        z.extractall(path=target)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--keep-archives", action="store_true", help="do not delete the .7z files")
    args = ap.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"models dir: {MODELS_DIR}")

    for asset, marker in ASSETS.items():
        marker_path = MODELS_DIR / marker
        if marker_path.exists() and not args.force:
            print(f"OK  {asset} already unpacked — skipping")
            continue

        archive = MODELS_DIR / asset
        if not archive.exists() or args.force:
            download(f"{BASE}/{asset}", archive)

        extract(archive, MODELS_DIR)

        if not marker_path.exists():
            print(f"ERROR: expected {marker_path} after extracting {asset}", file=sys.stderr)
            return 1

        if not args.keep_archives:
            archive.unlink()

    print("\nall weights ready:")
    for marker in ASSETS.values():
        print(f"  {MODELS_DIR / marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
