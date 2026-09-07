"""Extract two small BIRD databases and select five fixed smoke-test questions."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('unpacked_dev', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = root / 'data/bird'
    target.mkdir(parents=True, exist_ok=True)
    source = args.unpacked_dev / 'dev.json'
    rows = json.loads(source.read_text())
    rng = random.Random(42)
    selected = []
    for db, difficulty, count in [('superhero', 'simple', 2),
                                   ('student_club', 'simple', 1),
                                   ('student_club', 'moderate', 1),
                                   ('student_club', 'challenging', 1)]:
        pool = [r for r in rows if r['db_id'] == db and r['difficulty'] == difficulty]
        selected.extend(rng.sample(pool, count))
    selected.sort(key=lambda r: r['question_id'])
    selected_path = target / 'smoke5.json'
    if selected_path.exists() and json.loads(selected_path.read_text()) != selected:
        raise SystemExit('Existing smoke5.json differs; refusing to overwrite')
    selected_path.write_text(json.dumps(selected, indent=2, ensure_ascii=False) + '\n')
    dbs = {r['db_id'] for r in selected}
    with ZipFile(args.unpacked_dev / 'dev_databases.zip') as archive:
        for item in archive.infolist():
            parts = Path(item.filename).parts
            if len(parts) < 3 or parts[0] != 'dev_databases' or parts[1] not in dbs:
                continue
            if item.is_dir() or Path(item.filename).suffix not in ('.sqlite', '.csv'):
                continue
            output = (target / 'dev' / item.filename).resolve()
            if not output.is_relative_to((target / 'dev').resolve()):
                raise ValueError('Unsafe archive member')
            if not output.exists():
                output.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as src, output.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
    manifest = {'source': 'https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip',
                'archive_release': 'dev_20240627', 'seed': 42,
                'dev_json_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'question_ids': [r['question_id'] for r in selected],
                'note': 'Five-question engineering smoke test, not representative SDS.'}
    (target / 'smoke5_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
