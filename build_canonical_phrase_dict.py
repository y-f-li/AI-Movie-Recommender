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
    build_canonical_phrase_to_titles,
    build_canonical_title_to_qids,
    load_canonical_title_dict,
    save_canonical_phrase_dict,
)


def build(dataset_dir: Path, max_span_len: int = 5) -> Path:
    if max_span_len < 1:
        raise ValueError("max_span_len must be at least 1")

    titles_path = dataset_dir / "titles_to_qid.json"
    canonical_path = dataset_dir / "canonical_title_to_qids.json"

    if canonical_path.exists():
        canonical_map = load_canonical_title_dict(dataset_dir)
        canonical_source = canonical_path
    else:
        if not titles_path.exists():
            raise FileNotFoundError(f"Missing title dictionary: {titles_path}")
        with titles_path.open("r", encoding="utf-8") as f:
            title_to_qid = json.load(f)
        canonical_map = build_canonical_title_to_qids(title_to_qid)
        canonical_source = titles_path

    phrase_map = build_canonical_phrase_to_titles(canonical_map, max_span_len=max_span_len)
    out_path = save_canonical_phrase_dict(dataset_dir, phrase_map, max_span_len=max_span_len)

    print(f"Read canonical titles from: {canonical_source}")
    print(f"Wrote canonical phrase dictionary to: {out_path}")
    print(f"Canonical titles: {len(canonical_map)}")
    print(f"Distinct phrases: {len(phrase_map)}")
    multi = sum(1 for v in phrase_map.values() if len(v) > 1)
    print(f"Phrases mapping to multiple titles: {multi}")
    return out_path


if __name__ == "__main__":
    dataset = Path(os.getenv("DATAPATH", ROOT / "dataset")).expanduser().resolve()
    max_span_len = int(os.getenv("MAX_SPAN_LEN", "5"))
    build(dataset, max_span_len=max_span_len)
