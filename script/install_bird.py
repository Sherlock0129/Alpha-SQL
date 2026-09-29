"""Install official BIRD train/dev archives into the layout Alpha-SQL expects."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]


def _safe_extract(archive: ZipFile, destination: Path, strip_first: bool = False) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        parts = Path(member.filename).parts
        if strip_first:
            parts = parts[1:]
        if not parts:
            continue
        output = destination.joinpath(*parts).resolve()
        if not output.is_relative_to(destination):
            raise ValueError(f"Unsafe archive member: {member.filename}")
        if member.is_dir():
            output.mkdir(parents=True, exist_ok=True)
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, output.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def install(split: str, archive_path: Path, destination_root: Path) -> None:
    archive_path = archive_path.resolve()
    target = (destination_root / split).resolve()
    if target.exists():
        raise FileExistsError(f"Target already exists; refusing to overwrite: {target}")
    destination_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{split}-install-", dir=destination_root)).resolve()
    try:
        with ZipFile(archive_path) as archive:
            first_parts = {Path(item.filename).parts[0] for item in archive.infolist() if Path(item.filename).parts}
            strip_first = len(first_parts) == 1
            _safe_extract(archive, staging, strip_first=strip_first)

        # train.zip also contains a __MACOSX metadata root, so its real payload
        # remains under staging/train instead of being stripped above.
        content_root = staging / split if (staging / split).is_dir() else staging
        nested = content_root / f"{split}_databases.zip"
        if nested.exists():
            with ZipFile(nested) as database_archive:
                _safe_extract(database_archive, content_root)
            nested.unlink()

        data_file = content_root / f"{split}.json"
        database_dir = content_root / f"{split}_databases"
        if not data_file.exists() or not database_dir.exists():
            raise ValueError(f"Unexpected {split} archive layout")
        rows = json.loads(data_file.read_text(encoding="utf-8"))
        sqlite_files = list(database_dir.glob("*/*.sqlite"))
        db_ids = {str(row["db_id"]) for row in rows}
        installed_db_ids = {path.parent.name for path in sqlite_files}
        missing = sorted(db_ids - installed_db_ids)
        if missing:
            raise ValueError(f"Missing databases: {missing}")
        manifest = {
            "split": split,
            "source": f"https://bird-bench.oss-cn-beijing.aliyuncs.com/{split}.zip",
            "archive": str(archive_path),
            "archive_sha256": _sha256(archive_path),
            "question_count": len(rows),
            "database_count": len(db_ids),
            "sqlite_count": len(sqlite_files),
        }
        (content_root / "install_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if content_root == staging:
            staging.replace(target)
        else:
            content_root.replace(target)
            shutil.rmtree(staging, ignore_errors=True)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    except Exception:
        # The path is an exact mkdtemp child under destination_root, never a user-supplied broad path.
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("train", "dev"))
    parser.add_argument("archive", type=Path)
    parser.add_argument("--destination-root", type=Path, default=ROOT / "data" / "bird")
    arguments = parser.parse_args()
    install(arguments.split, arguments.archive, arguments.destination_root)
