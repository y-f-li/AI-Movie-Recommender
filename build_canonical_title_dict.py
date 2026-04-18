from __future__ import annotations

import json
import os
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.title_canonicalizer import (
    build_canonical_title_to_qids,
    save_canonical_title_dict,
    canonical_title_dict_path,
)


def build(dataset_dir: Path) -> Path:
    titles_path = dataset_dir / "titles_to_qid.json"
    if not titles_path.exists():
        raise FileNotFoundError(f"Missing title dictionary: {titles_path}")

    with titles_path.open("r", encoding="utf-8") as f:
        title_to_qid = json.load(f)

    canonical_map = build_canonical_title_to_qids(title_to_qid)
    out_path = save_canonical_title_dict(dataset_dir, canonical_map)

    print(f"Read titles from: {titles_path}")
    print(f"Wrote canonical title dictionary to: {out_path}")
    print(f"Raw titles: {len(title_to_qid)}")
    print(f"Canonical keys: {len(canonical_map)}")
    multi = sum(1 for v in canonical_map.values() if len(v) > 1)
    print(f"Canonical keys with multiple QIDs: {multi}")
    return out_path


if __name__ == "__main__":
    dataset = Path(os.getenv("DATAPATH", ROOT / "dataset")).expanduser().resolve()
    build(dataset)
