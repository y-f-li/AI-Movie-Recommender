from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

EXCLUDED_WORDS = {"the", "of", "a", "an", "and", "or", "film", "films", "movie", "movies"}
CARRIER_WORDS = {"film", "films", "movie", "movies"}


def _normalize(text: str) -> str:
    s = (text or "").strip().lower()
    s = s.replace("’", "'").replace("‘", "'").replace('“', '"').replace('”', '"')
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", _normalize(text)) if t and t not in EXCLUDED_WORDS]


def _strip_carrier_words(text: str) -> str:
    toks = [t for t in _tokenize(text) if t not in CARRIER_WORDS]
    return _normalize(" ".join(toks))


class EntitySynonymMatcher:
    """
    Deterministic entity synonym matcher backed by an offline-built artifact.

    The artifact stores:
      - word_to_synonyms: token -> expanded token list
      - alias_to_canonical: exact normalized alias phrase -> canonical label
      - canonical_labels: canonical label -> metadata incl. source_tokens and term_set
      - term_to_canonicals: token -> candidate canonical labels

    Runtime matching is intentionally conservative:
      1) exact label
      2) exact alias phrase
      3) scored synonym overlap over a shortlist from the inverted token index
    """

    def __init__(self, entity_id_dict: Dict[str, str], artifact_path: str | Path | None = None):
        self.entity_id_dict = {_normalize(k): v for k, v in (entity_id_dict or {}).items()}
        self.canonical_labels: Dict[str, dict] = {}
        self.word_to_synonyms: Dict[str, Set[str]] = {}
        self.alias_to_canonical: Dict[str, str] = {}
        self.term_to_canonicals: Dict[str, Set[str]] = {}
        self.artifact_loaded = False
        self.artifact_path = Path(artifact_path) if artifact_path else None

        if self.artifact_path and self.artifact_path.exists():
            try:
                with self.artifact_path.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
                self.canonical_labels = {
                    _normalize(k): {
                        "surface_forms": {_normalize(x) for x in v.get("surface_forms", []) if x},
                        "source_tokens": {_normalize(x) for x in v.get("source_tokens", []) if x},
                        "term_set": {_normalize(x) for x in v.get("term_set", []) if x},
                    }
                    for k, v in raw.get("canonical_labels", {}).items()
                }
                self.word_to_synonyms = {
                    _normalize(k): {_normalize(x) for x in vals if x}
                    for k, vals in raw.get("word_to_synonyms", {}).items()
                }
                self.alias_to_canonical = {
                    _normalize(alias): _normalize(canon)
                    for alias, canon in raw.get("alias_to_canonical", {}).items()
                }
                self.term_to_canonicals = {
                    _normalize(k): {_normalize(x) for x in vals if x}
                    for k, vals in raw.get("term_to_canonicals", {}).items()
                }
                self.artifact_loaded = True
            except Exception:
                self.artifact_loaded = False

        # Backstop so canonical exact labels are always available.
        for label in self.entity_id_dict.keys():
            self.alias_to_canonical.setdefault(label, label)

    def _expand_query_tokens(self, tokens: List[str]) -> Tuple[Set[str], Dict[str, Set[str]]]:
        expanded_terms: Set[str] = set()
        token_synsets: Dict[str, Set[str]] = {}
        for token in tokens:
            syns = set(self.word_to_synonyms.get(token, set()))
            syns.add(token)
            token_synsets[token] = syns
            expanded_terms.update(syns)
        return expanded_terms, token_synsets

    def _score_candidate(self, query_norm: str, query_tokens: List[str], expanded_terms: Set[str], token_synsets: Dict[str, Set[str]], canonical: str) -> float:
        meta = self.canonical_labels.get(canonical)
        if not meta:
            return 0.0

        source_tokens = set(meta.get("source_tokens") or set())
        term_set = set(meta.get("term_set") or set())
        surface_forms = set(meta.get("surface_forms") or set())
        cand_tokens = set(_tokenize(canonical))

        if query_norm == canonical or query_norm in surface_forms:
            return 1.0

        if not query_tokens:
            return 0.0

        covered = 0.0
        for token in query_tokens:
            syns = token_synsets.get(token, {token})
            if syns & source_tokens:
                covered += 1.0
        token_syn_coverage = covered / max(len(query_tokens), 1)

        exact_token_overlap = 0.0
        q_token_set = set(query_tokens)
        if cand_tokens:
            exact_token_overlap = len(q_token_set & cand_tokens) / max(len(q_token_set), len(cand_tokens))

        expanded_overlap = 0.0
        if term_set and expanded_terms:
            expanded_overlap = len(expanded_terms & term_set) / max(len(expanded_terms), len(term_set))

        containment_bonus = 1.0 if q_token_set and q_token_set.issubset(term_set | source_tokens | cand_tokens) else 0.0

        return 0.60 * token_syn_coverage + 0.20 * exact_token_overlap + 0.10 * expanded_overlap + 0.10 * containment_bonus

    def resolve_label(self, text: str, min_score: float = 0.60) -> Tuple[str | None, float]:
        query_norm = _normalize(text)
        if not query_norm:
            return None, 0.0

        stripped_norm = _strip_carrier_words(query_norm)

        for probe in [query_norm, stripped_norm]:
            if not probe:
                continue
            if probe in self.entity_id_dict:
                return probe, 1.0
            if probe in self.alias_to_canonical:
                return self.alias_to_canonical[probe], 1.0

        if not self.artifact_loaded:
            return None, 0.0

        query_tokens = _tokenize(stripped_norm or query_norm)
        if not query_tokens:
            return None, 0.0
        expanded_terms, token_synsets = self._expand_query_tokens(query_tokens)

        candidate_labels: Set[str] = set()
        for term in expanded_terms:
            candidate_labels.update(self.term_to_canonicals.get(term, set()))
        if not candidate_labels:
            return None, 0.0

        scored = []
        for canonical in candidate_labels:
            score = self._score_candidate(query_norm, query_tokens, expanded_terms, token_synsets, canonical)
            if score > 0.0:
                scored.append((score, canonical))
        if not scored:
            return None, 0.0

        scored.sort(key=lambda x: (x[0], len(_tokenize(x[1]))), reverse=True)
        best_score, best_label = scored[0]
        if best_score < min_score:
            return None, best_score
        return best_label, best_score

    def match_labels_from_text(self, text: str, max_ngram: int = 4, min_score: float = 0.60) -> List[str]:
        query_norm = _normalize(text)
        if not query_norm:
            return []

        label, score = self.resolve_label(query_norm, min_score=min_score)
        return [label] if label else []
