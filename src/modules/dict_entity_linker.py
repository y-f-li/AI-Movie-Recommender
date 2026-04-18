from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


EXCLUDED_WORDS = {"the", "of", "a", "an"}


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[A-Za-z0-9]+", (text or "").lower()) if t not in EXCLUDED_WORDS}


def _normalize(text: str) -> str:
    s = (text or "").strip().lower()
    s = s.replace("’", "'").replace("‘", "'").replace('“', '"').replace('”', '"')
    s = re.sub(r"\s+", " ", s)
    return s


@dataclass
class Candidate:
    label: str
    iri: str
    kind: str  # movie | entity


class DictEntityLinker:
    """
    Lightweight dictionary-based replacement for the old graph-backed EntityLinker.

    Prefers exact matches. Falls back to fuzzy/token overlap matching across
    prebuilt movie and entity label dictionaries.
    """

    def __init__(self, movie_id_dict: Dict[str, str], entity_id_dict: Dict[str, str], entity_synonym_matcher=None):
        self.movie_id_dict = {_normalize(k): v for k, v in movie_id_dict.items()}
        self.entity_id_dict = {_normalize(k): v for k, v in entity_id_dict.items()}
        self.entity_synonym_matcher = entity_synonym_matcher

        self._candidates: List[Candidate] = []
        for label, iri in self.movie_id_dict.items():
            self._candidates.append(Candidate(label=label, iri=iri, kind="movie"))
        for label, iri in self.entity_id_dict.items():
            # avoid duplicate exact labels already covered by movie titles
            if label not in self.movie_id_dict:
                self._candidates.append(Candidate(label=label, iri=iri, kind="entity"))

    def find_entity_by_label(self, entity_label: str) -> Tuple[str | None, str | None, bool | None]:
        query = _normalize(entity_label)
        if not query:
            return None, None, None

        # Exact movie match first, then entity exact.
        if query in self.movie_id_dict:
            return self.movie_id_dict[query], query, False
        if query in self.entity_id_dict:
            return self.entity_id_dict[query], query, False
        if self.entity_synonym_matcher is not None:
            canonical, score = self.entity_synonym_matcher.resolve_label(query)
            if canonical and canonical in self.entity_id_dict:
                return self.entity_id_dict[canonical], canonical, False

        query_tokens = _tokenize(query)
        scored: List[Tuple[float, Candidate]] = []

        for cand in self._candidates:
            ratio = difflib.SequenceMatcher(None, query, cand.label).ratio()
            token_overlap = 0.0
            cand_tokens = _tokenize(cand.label)
            if query_tokens and cand_tokens:
                token_overlap = len(query_tokens & cand_tokens) / max(len(query_tokens), len(cand_tokens))

            # Slightly prefer movies for ambiguous labels in movie-focused questions.
            score = ratio * 0.75 + token_overlap * 0.25 + (0.02 if cand.kind == "movie" else 0.0)
            scored.append((score, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return None, None, None

        best_score, best = scored[0]
        if best_score < 0.55:
            return None, None, None

        return best.iri, best.label, True
