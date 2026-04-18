from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Set

PREFIX_WD = "http://www.wikidata.org/entity/"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
P31 = "http://www.wikidata.org/prop/direct/P31"

LABEL_RE = re.compile(
    r'^<(?P<subj>[^>]+)>\s+<%s>\s+"(?P<label>(?:[^"\\]|\\.)*)"(?:@(?P<lang>[A-Za-z-]+)|\^\^<[^>]+>)?\s+\.$'
    % re.escape(RDFS_LABEL)
)
URI_OBJ_RE = re.compile(r'^<(?P<subj>[^>]+)>\s+<(?P<pred>[^>]+)>\s+<(?P<obj>[^>]+)>\s+\.$')


def _decode_nt_string(value: str) -> str:
    try:
        return bytes(value, "utf-8").decode("unicode_escape")
    except Exception:
        return value


def iri_to_qid(iri: str) -> str:
    iri = str(iri)
    if iri.startswith(PREFIX_WD):
        return iri[len(PREFIX_WD):]
    return iri.rsplit("/", 1)[-1]


def build(dataset_dir: Path) -> Path:
    graph_path = dataset_dir / "graph.nt"
    if not graph_path.exists():
        raise FileNotFoundError(f"Missing graph file: {graph_path}")

    labels_by_subject: Dict[str, str] = {}
    types_by_subject: DefaultDict[str, Set[str]] = defaultdict(set)

    with graph_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            m_label = LABEL_RE.match(line)
            if m_label:
                lang = (m_label.group("lang") or "").lower()
                if lang in {"", "en"}:
                    subj = m_label.group("subj")
                    label = _decode_nt_string(m_label.group("label")).strip()
                    if label:
                        labels_by_subject.setdefault(subj, label.lower())
                continue

            m_uri = URI_OBJ_RE.match(line)
            if not m_uri:
                continue

            if m_uri.group("pred") == P31:
                types_by_subject[m_uri.group("subj")].add(m_uri.group("obj"))

    film_class_iris: Set[str] = {
        subj for subj, label in labels_by_subject.items() if label == "film"
    }
    if not film_class_iris:
        raise RuntimeError(
            "Could not find any graph entity labeled exactly 'film'. "
            "The strict graph-only builder has no class to anchor on."
        )

    title_to_qids: DefaultDict[str, list[str]] = defaultdict(list)
    candidate_subjects = 0

    for subj, subj_types in types_by_subject.items():
        if not (subj_types & film_class_iris):
            continue
        title = labels_by_subject.get(subj)
        if not title:
            continue
        qid = iri_to_qid(subj)
        bucket = title_to_qids[title]
        if qid not in bucket:
            bucket.append(qid)
            candidate_subjects += 1

    for title, qids in title_to_qids.items():
        qids.sort()

    result = dict(sorted(title_to_qids.items()))
    out_path = dataset_dir / "titles_to_qids.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True)

    duplicate_titles = sum(1 for qids in result.values() if len(qids) > 1)
    max_qids = max((len(qids) for qids in result.values()), default=0)

    print(f"Read graph from: {graph_path}")
    print("Builder mode: strict graph-only")
    print("Inclusion rule: subject must have direct P31 -> entity whose label is exactly 'film'")
    print(f"Detected film class IRIs: {len(film_class_iris)}")
    print(f"Candidate film subjects written: {candidate_subjects}")
    print(f"Wrote multi-title dictionary to: {out_path}")
    print(f"Unique title keys: {len(result)}")
    print(f"Titles with multiple QIDs: {duplicate_titles}")
    print(f"Largest QID bucket size: {max_qids}")
    if "titanic" in result:
        print(f"Titanic bucket: {result['titanic']}")
    else:
        print("Titanic bucket: <missing>")

    return out_path


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent
    dataset = Path(os.getenv("DATAPATH", ROOT / "dataset")).expanduser().resolve()
    build(dataset)
