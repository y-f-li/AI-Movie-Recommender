from __future__ import annotations

import ast
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.paths import ensure_datapath, get_runtime_artifacts_dir

GENRE_RELATION = 'http://www.wikidata.org/prop/direct/P136'
ATAI_TAG_RELATION = 'http://ddis.ch/atai/tag'
TARGET_ATTRS = {GENRE_RELATION, ATAI_TAG_RELATION}
CARRIER_WORDS = {'film', 'films', 'movie', 'movies'}
STOP_WORDS = {'the', 'of', 'a', 'an', 'and', 'or'}
RDFS_LABEL = 'http://www.w3.org/2000/01/rdf-schema#label'


def normalize(text: str) -> str:
    s = (text or '').strip().lower()
    s = s.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')
    s = s.replace('_', ' ')
    s = s.replace('-', ' ')
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r'[a-z0-9]+', normalize(text)) if t and t not in STOP_WORDS and t not in CARRIER_WORDS]


def carrier_stripped_phrase(text: str) -> str:
    return normalize(' '.join(tokenize(text)))


def normalize_value_key(value: str) -> str:
    s = str(value).strip()
    if s.startswith('http://') or s.startswith('https://'):
        return s.rstrip('/').split('/')[-1].strip().lower()
    return normalize(s)


def find_graph_path(dataset_dir: Path) -> Path:
    candidates = [
        dataset_dir / 'graph.nt',
        ROOT / 'dataset' / 'graph.nt',
        ROOT / 'graph.nt',
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        'Could not find graph.nt. Expected one of: ' + ', '.join(str(p) for p in candidates)
    )


def _parse_nt_subject_and_predicate(line: str) -> tuple[Optional[str], Optional[str], str]:
    if not line.startswith('<'):
        return None, None, ''
    try:
        s_end = line.index('>')
        subj = line[1:s_end]
        rest = line[s_end + 1:].lstrip()
        if not rest.startswith('<'):
            return None, None, ''
        p_end = rest.index('>')
        pred = rest[1:p_end]
        obj = rest[p_end + 1:].lstrip()
        return subj, pred, obj
    except ValueError:
        return None, None, ''


def _parse_nt_literal(obj: str) -> tuple[Optional[str], Optional[str]]:
    if not obj.startswith('"'):
        return None, None
    escaped = False
    end_idx = None
    for i in range(1, len(obj)):
        ch = obj[i]
        if escaped:
            escaped = False
            continue
        if ch == '\\':
            escaped = True
            continue
        if ch == '"':
            end_idx = i
            break
    if end_idx is None:
        return None, None
    lit_src = obj[: end_idx + 1]
    tail = obj[end_idx + 1 :].strip()
    try:
        value = ast.literal_eval(lit_src)
    except Exception:
        value = lit_src[1:-1].encode('utf-8', 'ignore').decode('unicode_escape', 'ignore')
    lang = None
    if tail.startswith('@'):
        lang = tail[1:].split()[0].lower()
    return value, lang


def _label_rank(lang: Optional[str]) -> int:
    if lang == 'en':
        return 3
    if lang is None:
        return 2
    return 1


def load_graph_labels(graph_path: Path, needed_iris: Set[str]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    best_rank: Dict[str, int] = {}
    with graph_path.open('r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            if RDFS_LABEL not in line:
                continue
            subj, pred, obj = _parse_nt_subject_and_predicate(line)
            if pred != RDFS_LABEL or subj not in needed_iris:
                continue
            value, lang = _parse_nt_literal(obj)
            if not value:
                continue
            rank = _label_rank(lang)
            if rank >= best_rank.get(subj, -1):
                labels[subj] = str(value)
                best_rank[subj] = rank
    return labels


def build(dataset_dir: Path):
    artifacts_dir = get_runtime_artifacts_dir()
    attr_index_path = artifacts_dir / 'attr_index.pkl'

    if not attr_index_path.exists():
        raise FileNotFoundError(f'Missing attr_index artifact: {attr_index_path}')

    with attr_index_path.open('rb') as f:
        attr_index = pickle.load(f)

    needed_iris: Set[str] = set()
    for attr in TARGET_ATTRS:
        value_to_movies = attr_index.get(attr, {})
        for raw_value in value_to_movies.keys():
            raw_str = str(raw_value)
            if raw_str.startswith('http://') or raw_str.startswith('https://'):
                needed_iris.add(raw_str)

    graph_path = find_graph_path(dataset_dir)
    graph_labels = load_graph_labels(graph_path, needed_iris)

    output = {'version': 3, 'source': 'graph.nt', 'graph_path': str(graph_path), 'attrs': {}}

    for attr in TARGET_ATTRS:
        value_to_movies = attr_index.get(attr, {})
        values_meta: Dict[str, dict] = {}
        phrase_lookup: Dict[str, Set[str]] = defaultdict(set)
        token_lookup: Dict[str, Set[str]] = defaultdict(set)
        lookup: Dict[str, Set[str]] = defaultdict(set)

        for raw_value in sorted(value_to_movies.keys(), key=lambda x: str(x)):
            raw_value_str = str(raw_value)
            if raw_value_str.startswith('http://') or raw_value_str.startswith('https://'):
                label = str(graph_labels.get(raw_value_str, raw_value_str) or raw_value_str)
            else:
                label = raw_value_str
            norm_label = normalize(label)
            stripped_phrase = carrier_stripped_phrase(label)
            core_terms: List[str] = sorted(set(tokenize(label)))

            values_meta[raw_value_str] = {
                'label': label,
                'normalized_label': norm_label,
                'stripped_phrase': stripped_phrase,
                'core_terms': core_terms,
            }

            if stripped_phrase:
                phrase_lookup[stripped_phrase].add(raw_value_str)
                lookup[stripped_phrase].add(raw_value_str)
            for term in core_terms:
                token_lookup[term].add(raw_value_str)
                lookup[term].add(raw_value_str)

            for key in {raw_value_str, normalize(raw_value_str), normalize_value_key(raw_value_str), norm_label}:
                if key:
                    lookup[key].add(raw_value_str)

        output['attrs'][attr] = {
            'values': values_meta,
            'phrase_lookup': {k: sorted(v) for k, v in sorted(phrase_lookup.items())},
            'token_lookup': {k: sorted(v) for k, v in sorted(token_lookup.items())},
            'lookup': {k: sorted(v) for k, v in sorted(lookup.items())},
        }

    out_path = artifacts_dir / 'genre_tag_bag_index.json'
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False)

    print('Wrote genre/tag bag artifact:')
    print(' -', out_path)
    print('Source graph:', graph_path)
    print('Graph labels found:', len(graph_labels), '/', len(needed_iris), 'IRI values')
    for attr, data in output['attrs'].items():
        print(f'[{attr}] values:', len(data['values']), 'phrase keys:', len(data['phrase_lookup']), 'token keys:', len(data['token_lookup']))

    thriller_key = 'thriller'
    print(f'\nValues indexed under token key {thriller_key!r}:')
    for attr in TARGET_ATTRS:
        attr_data = output['attrs'].get(attr, {})
        vals = attr_data.get('token_lookup', {}).get(thriller_key, [])
        print(f'[{attr}] {len(vals)} values')
        for raw_value in vals:
            meta = attr_data.get('values', {}).get(raw_value, {})
            print(' -', meta.get('label', raw_value), '::', raw_value)


if __name__ == '__main__':
    dataset = ensure_datapath()
    build(dataset)
