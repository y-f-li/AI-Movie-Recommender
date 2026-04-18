from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modules.title_canonicalizer import (
    build_canonical_title_to_qids,
    build_det_stripped_title_to_titles,
    canonical_title_dict_path,
    det_stripped_title_dict_path,
    load_canonical_title_dict,
    save_det_stripped_title_dict,
)


def main() -> int:
    dataset_dir = Path(os.getenv('DATAPATH', ROOT / 'dataset')).expanduser().resolve()
    canonical_path = canonical_title_dict_path(dataset_dir)
    if canonical_path.exists():
        canonical_map = load_canonical_title_dict(dataset_dir)
    else:
        titles_to_qid_path = dataset_dir / 'titles_to_qid.json'
        if not titles_to_qid_path.exists():
            raise FileNotFoundError(
                f"Could not find {canonical_path.name} or titles_to_qid.json in {dataset_dir}"
            )
        with titles_to_qid_path.open('r', encoding='utf-8') as f:
            title_to_qid = json.load(f)
        canonical_map = build_canonical_title_to_qids({str(k): str(v) for k, v in title_to_qid.items()})

    det_map = build_det_stripped_title_to_titles(canonical_map)
    out_path = save_det_stripped_title_dict(dataset_dir, det_map)
    print(f"Wrote {out_path}")
    print(f"canonical titles: {len(canonical_map)}")
    print(f"det-stripped aliases: {len(det_map)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
