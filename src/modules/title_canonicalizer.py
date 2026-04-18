from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List


SMART_TRANSLATION = str.maketrans({
    "’": "'",
    "‘": "'",
    '“': '"',
    '”': '"',
    "–": "-",
    "—": "-",
    "…": " ",
})

DET_STRIP_WORDS = {"the", "a", "an"}


def strip_det_tokens_from_canonical_text(text: str) -> str:
    canonical = canonicalize_title_text(text)
    if not canonical:
        return ""
    tokens = [tok for tok in canonical.split() if tok not in DET_STRIP_WORDS]
    return " ".join(tokens).strip()


def build_det_stripped_title_to_titles(canonical_title_to_qids: Dict[str, List[str]]) -> Dict[str, List[str]]:
    det_map: Dict[str, List[str]] = {}
    for canonical_title in canonical_title_to_qids.keys():
        key = strip_det_tokens_from_canonical_text(canonical_title)
        if not key:
            continue
        bucket = det_map.setdefault(key, [])
        if canonical_title not in bucket:
            bucket.append(canonical_title)
    for key in det_map:
        det_map[key].sort()
    return det_map


def det_stripped_title_dict_path(dataset_dir: str | os.PathLike[str]) -> Path:
    return Path(dataset_dir) / "det_stripped_title_to_titles.json"


def save_det_stripped_title_dict(dataset_dir: str | os.PathLike[str], det_map: Dict[str, List[str]]) -> Path:
    path = det_stripped_title_dict_path(dataset_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(det_map, f, ensure_ascii=False, sort_keys=True)
    return path


def load_det_stripped_title_dict(dataset_dir: str | os.PathLike[str]) -> Dict[str, List[str]]:
    path = det_stripped_title_dict_path(dataset_dir)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    normalized: Dict[str, List[str]] = {}
    for key, value in data.items():
        if isinstance(value, list):
            normalized[str(key)] = [str(v) for v in value]
        else:
            normalized[str(key)] = [str(value)]
    return normalized



def canonicalize_title_text(text: str) -> str:
    s = (text or "").strip().translate(SMART_TRANSLATION).lower()
    # turn any run of non-alphanumeric chars into a single space
    s = re.sub(r"[^0-9a-z]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def build_canonical_title_to_qids(title_to_qid: Dict[str, str]) -> Dict[str, List[str]]:
    canonical: Dict[str, List[str]] = {}
    for raw_title, qid in title_to_qid.items():
        key = canonicalize_title_text(raw_title)
        if not key:
            continue
        qid_str = str(qid)
        bucket = canonical.setdefault(key, [])
        if qid_str not in bucket:
            bucket.append(qid_str)
    return canonical


def canonical_title_dict_path(dataset_dir: str | os.PathLike[str]) -> Path:
    return Path(dataset_dir) / "canonical_title_to_qids.json"


def save_canonical_title_dict(dataset_dir: str | os.PathLike[str], canonical_map: Dict[str, List[str]]) -> Path:
    path = canonical_title_dict_path(dataset_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(canonical_map, f, ensure_ascii=False, sort_keys=True)
    return path


def load_canonical_title_dict(dataset_dir: str | os.PathLike[str]) -> Dict[str, List[str]]:
    path = canonical_title_dict_path(dataset_dir)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # normalize shape defensively
    normalized: Dict[str, List[str]] = {}
    for key, value in data.items():
        if isinstance(value, list):
            normalized[str(key)] = [str(v) for v in value]
        else:
            normalized[str(key)] = [str(value)]
    return normalized



def build_canonical_phrase_to_titles(canonical_title_to_qids: Dict[str, List[str]], max_span_len: int = 5) -> Dict[str, List[str]]:
    if max_span_len < 1:
        raise ValueError("max_span_len must be at least 1")

    phrase_map: Dict[str, List[str]] = {}
    for canonical_title in canonical_title_to_qids.keys():
        tokens = canonical_title.split()
        if not tokens:
            continue
        n = len(tokens)
        for start in range(n):
            max_len_here = min(max_span_len, n - start)
            for span_len in range(1, max_len_here + 1):
                phrase = " ".join(tokens[start:start + span_len])
                if not phrase:
                    continue
                bucket = phrase_map.setdefault(phrase, [])
                if canonical_title not in bucket:
                    bucket.append(canonical_title)

    for phrase in phrase_map:
        phrase_map[phrase].sort()
    return phrase_map


def canonical_phrase_dict_path(dataset_dir: str | os.PathLike[str], max_span_len: int = 5) -> Path:
    return Path(dataset_dir) / f"canonical_phrase_to_titles_up_to_{max_span_len}.json"


def save_canonical_phrase_dict(
    dataset_dir: str | os.PathLike[str],
    phrase_map: Dict[str, List[str]],
    max_span_len: int = 5,
) -> Path:
    path = canonical_phrase_dict_path(dataset_dir, max_span_len=max_span_len)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(phrase_map, f, ensure_ascii=False, sort_keys=True)
    return path


def load_canonical_phrase_dict(
    dataset_dir: str | os.PathLike[str],
    max_span_len: int = 5,
) -> Dict[str, List[str]]:
    path = canonical_phrase_dict_path(dataset_dir, max_span_len=max_span_len)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    normalized: Dict[str, List[str]] = {}
    for key, value in data.items():
        if isinstance(value, list):
            normalized[str(key)] = [str(v) for v in value]
        else:
            normalized[str(key)] = [str(value)]
    return normalized
