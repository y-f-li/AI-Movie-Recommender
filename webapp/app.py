from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, render_template, request
import re
from difflib import SequenceMatcher

try:
    from rapidfuzz import fuzz, process
except Exception:
    fuzz = None
    process = None

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modules.config import Config
from modules.agent import Agent
from modules.runtime.artifact_store import RuntimeArtifactStore
from modules.title_canonicalizer import (
    build_canonical_phrase_to_titles,
    build_canonical_title_to_qids,
    build_det_stripped_title_to_titles,
    canonical_phrase_dict_path,
    canonicalize_title_text,
    canonical_title_dict_path,
    det_stripped_title_dict_path,
    load_canonical_phrase_dict,
    load_canonical_title_dict,
    load_det_stripped_title_dict,
    strip_det_tokens_from_canonical_text,
)

from modules.paths import ensure_datapath, get_dataset_dir

DEFAULT_DATASET_DIR = get_dataset_dir()
DEFAULT_ARTIFACTS_DIR = DEFAULT_DATASET_DIR / "runtime_artifacts"

app = Flask(__name__, template_folder="templates", static_folder="static")

_agent_lock = threading.Lock()
_agent: Optional[Agent] = None
_agent_error: Optional[str] = None


def _build_config(dataset_dir: Path) -> Config:
    return Config(
        kb_path="",
        ratings_path=str(dataset_dir / "ratings"),
        host_url="local-webapp",
        username="local-webapp",
        password="local-webapp",
        artifacts_dir=str(dataset_dir / "runtime_artifacts"),
    )


def get_agent() -> Optional[Agent]:
    global _agent, _agent_error
    if _agent is not None or _agent_error is not None:
        return _agent

    with _agent_lock:
        if _agent is not None or _agent_error is not None:
            return _agent
        try:
            dataset_dir = ensure_datapath()
            _agent = Agent(_build_config(dataset_dir))
        except Exception as exc:
            _agent_error = f"{type(exc).__name__}: {exc}"
            _agent = None
    return _agent


@app.get("/")
def index():
    return render_template("index.html")


def _load_canonical_title_map(resolver) -> dict[str, list[str]]:
    path = canonical_title_dict_path(DEFAULT_DATASET_DIR)
    if path.exists():
        try:
            return load_canonical_title_dict(DEFAULT_DATASET_DIR)
        except Exception:
            pass
    return build_canonical_title_to_qids(resolver.t_q_dict)

def _load_canonical_phrase_map(resolver, max_span_len: int = 5) -> dict[str, list[str]]:
    path = canonical_phrase_dict_path(DEFAULT_DATASET_DIR, max_span_len=max_span_len)
    if path.exists():
        try:
            return load_canonical_phrase_dict(DEFAULT_DATASET_DIR, max_span_len=max_span_len)
        except Exception:
            pass
    canonical_title_map = _load_canonical_title_map(resolver)
    return build_canonical_phrase_to_titles(canonical_title_map, max_span_len=max_span_len)


def _load_det_stripped_title_map(resolver) -> dict[str, list[str]]:
    path = det_stripped_title_dict_path(DEFAULT_DATASET_DIR)
    if path.exists():
        try:
            return load_det_stripped_title_dict(DEFAULT_DATASET_DIR)
        except Exception:
            pass
    canonical_title_map = _load_canonical_title_map(resolver)
    return build_det_stripped_title_to_titles(canonical_title_map)


def _load_title_to_qids_map(_resolver=None) -> tuple[dict[str, list[str]], dict[str, object]]:
    """
    Strict loader for the multi-QID title dictionary.

    This loader intentionally trusts only dataset/titles_to_qids.json.
    It does NOT fall back to runtime_artifacts/id_movie.json or the old
    single-QID resolver map, because those can silently collapse or bias
    ambiguous titles such as "Titanic".
    """
    titles_to_qids_path = DEFAULT_DATASET_DIR / "titles_to_qids.json"
    meta: dict[str, object] = {
        "source": str(titles_to_qids_path),
        "strict": True,
        "exists": titles_to_qids_path.exists(),
        "loaded": False,
        "error": None,
    }

    if not titles_to_qids_path.exists():
        meta["error"] = f"Missing file: {titles_to_qids_path.name}"
        return {}, meta

    try:
        with titles_to_qids_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
        return {}, meta

    result: dict[str, list[str]] = {}
    for raw_title, raw_qids in raw.items():
        title = str(raw_title).strip().lower()
        if not title:
            continue
        qids = raw_qids if isinstance(raw_qids, list) else [raw_qids]
        cleaned: list[str] = []
        for qid in qids:
            qid_str = str(qid).strip()
            if not qid_str:
                continue
            if not qid_str.startswith("Q"):
                qid_str = qid_str.rsplit("/", 1)[-1]
            if qid_str and qid_str not in cleaned:
                cleaned.append(qid_str)
        if cleaned:
            result[title] = sorted(cleaned)

    meta["loaded"] = True
    meta["title_count"] = len(result)
    return result, meta


def _build_title_token_choices(canonical_title_map: dict[str, list[str]], det_stripped_title_map: dict[str, list[str]]) -> list[str]:
    tokens: set[str] = set()
    for title in canonical_title_map.keys():
        tokens.update(tok for tok in title.split() if tok)
    for title in det_stripped_title_map.keys():
        tokens.update(tok for tok in title.split() if tok)
    return sorted(tokens)


def _normalized_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if process is not None and fuzz is not None:
        return float(fuzz.ratio(a, b))
    return 100.0 * SequenceMatcher(None, a, b).ratio()


def _extract_fuzzy_choices(query: str, choices: list[str], *, limit: int = 8, score_cutoff: float = 88.0) -> list[tuple[str, float]]:
    if not query or not choices:
        return []
    if process is not None and fuzz is not None:
        matches = process.extract(query, choices, scorer=fuzz.WRatio, limit=limit, score_cutoff=score_cutoff)
        return [(str(choice), float(score)) for choice, score, _idx in matches]

    scored: list[tuple[str, float]] = []
    for choice in choices:
        score = _normalized_similarity(query, choice)
        if score >= score_cutoff:
            scored.append((choice, score))
    scored.sort(key=lambda item: (-item[1], -len(item[0].split()), -len(item[0]), item[0]))
    return scored[:limit]


def _detect_typo_suspicions_in_text(text: str, title_token_choices: list[str]) -> list[dict]:
    suspicions: list[dict] = []
    for token in canonicalize_title_text(text).split():
        if not token or token.isdigit() or len(token) <= 2:
            continue
        if token in title_token_choices:
            continue
        suggestions = _extract_fuzzy_choices(token, title_token_choices, limit=1, score_cutoff=80.0)
        if not suggestions:
            continue
        suggestion, score = suggestions[0]
        suspicions.append({
            "token": token,
            "suggestion": suggestion,
            "score": round(float(score), 2),
        })
    return suspicions


def _resolve_raw_title_for_canonical_qid(resolver, canonical_title: str, qid_short: str) -> str:
    for raw_title, raw_qid in resolver.t_q_dict.items():
        raw_qid_short = raw_qid if str(raw_qid).startswith("Q") else str(raw_qid).rsplit("/", 1)[-1]
        if raw_qid_short == qid_short and canonicalize_title_text(raw_title) == canonical_title:
            return raw_title
    return canonical_title


def _find_aligned_candidate_match(candidate_title: str, anchor_phrase: str, anchor_start: int, anchor_end: int, original_tokens: list[str], consumed_mask: list[bool]) -> dict | None:
    candidate_tokens = candidate_title.split()
    anchor_tokens = anchor_phrase.split()
    if not candidate_tokens or not anchor_tokens:
        return None

    anchor_len = len(anchor_tokens)
    for cand_anchor_start in range(0, len(candidate_tokens) - anchor_len + 1):
        if candidate_tokens[cand_anchor_start:cand_anchor_start + anchor_len] != anchor_tokens:
            continue
        message_start = anchor_start - cand_anchor_start
        message_end = message_start + len(candidate_tokens) - 1
        if message_start < 0 or message_end >= len(original_tokens):
            continue
        if any(consumed_mask[i] for i in range(message_start, message_end + 1)):
            continue
        if original_tokens[message_start:message_end + 1] == candidate_tokens:
            return {
                "message_start": message_start,
                "message_end": message_end,
                "candidate_anchor_start": cand_anchor_start,
                "candidate_anchor_end": cand_anchor_start + anchor_len - 1,
                "candidate_tokens": candidate_tokens,
            }
    return None

def _normalize_lookup_text(text: str) -> str:
    s = (text or "").strip().lower()
    s = s.replace("’", "'").replace("‘", "'").replace('“', '"').replace('”', '"')
    s = re.sub(r"\s+", " ", s)
    return s


def _normalize_pos_debug_text(text: str) -> str:
    s = (text or "").strip()
    s = s.replace("’", "'").replace("‘", "'").replace('“', '"').replace('”', '"')
    s = re.sub(r"\s+", " ", s)
    return s


def _normalize_span_text(text: str) -> str:
    s = _normalize_lookup_text(text)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"([([\{])\s+", r"\1", s)
    s = re.sub(r"\s+([)\]\}])", r"\1", s)
    s = re.sub(r"\s+n't\b", "n't", s)
    s = re.sub(r"\s+'(s|re|ve|d|ll|m)\b", r"'\1", s)
    return re.sub(r"\s+", " ", s).strip()


def _rebuild_query_from_tokens(tokens: list[str]) -> str:
    s = " ".join(tokens)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"([([\{])\s+", r"\1", s)
    s = re.sub(r"\s+([)\]\}])", r"\1", s)
    s = re.sub(r"\s+n't\b", "n't", s)
    s = re.sub(r"\s+'(s|re|ve|d|ll|m)\b", r"'\1", s)
    return re.sub(r"\s+", " ", s).strip()

def _build_visible_tokens_with_mapping(original_tokens: list[str], consumed_mask: list[bool], placeholder: str) -> tuple[list[str], dict[int, int]]:
    visible_tokens: list[str] = []
    original_to_visible: dict[int, int] = {}
    idx = 0
    while idx < len(original_tokens):
        if consumed_mask[idx]:
            visible_tokens.append(placeholder)
            while idx < len(original_tokens) and consumed_mask[idx]:
                idx += 1
        else:
            original_to_visible[idx] = len(visible_tokens)
            visible_tokens.append(original_tokens[idx])
            idx += 1
    return visible_tokens, original_to_visible


def _build_placeholder_view_from_consumed(original_tokens: list[str], consumed_mask: list[bool], placeholder: str) -> str:
    visible_tokens, _ = _build_visible_tokens_with_mapping(original_tokens, consumed_mask, placeholder)
    return _rebuild_query_from_tokens(visible_tokens)


_TITLE_CONTEXT_PATTERNS: list[tuple[list[str], str]] = [
    (["movies", "like"], "movies like"),
    (["films", "like"], "films like"),
    (["something", "like"], "something like"),
    (["similar", "to"], "similar to"),
    (["the", "movie"], "the movie"),
    (["the", "film"], "the film"),
    (["a", "movie", "called"], "a movie called"),
    (["the", "movie", "called"], "the movie called"),
    (["a", "film", "called"], "a film called"),
    (["the", "film", "called"], "the film called"),
    (["the", "movie", "is", "called"], "the movie is called"),
    (["the", "film", "is", "called"], "the film is called"),
    (["the", "name", "of", "the", "movie", "is"], "the name of the movie is"),
    (["the", "name", "of", "the", "film", "is"], "the name of the film is"),
    (["the", "title", "is"], "the title is"),
    (["the", "movie", "titled"], "the movie titled"),
    (["the", "film", "titled"], "the film titled"),
    (["i", "like"], "i like"),
    (["i", "love"], "i love"),
    (["i", "loved"], "i loved"),
    (["i", "enjoyed"], "i enjoyed"),
    (["i", "watched"], "i watched"),
    (["i", "saw"], "i saw"),
    (["i", "just", "watched"], "i just watched"),
    (["i", "just", "saw"], "i just saw"),
    (["i", "recently", "watched"], "i recently watched"),
    (["i", "recently", "saw"], "i recently saw"),
]


def _contains_consecutive_pattern(tokens: list[str], pattern: list[str]) -> bool:
    if not pattern or len(tokens) < len(pattern):
        return False
    for start in range(0, len(tokens) - len(pattern) + 1):
        if tokens[start:start + len(pattern)] == pattern:
            return True
    return False


def _prefix_has_title_trigger(prefix_tokens: list[str]) -> bool:
    for pattern, _label in _TITLE_CONTEXT_PATTERNS:
        if _contains_consecutive_pattern(prefix_tokens, pattern):
            return True
    for idx, token in enumerate(prefix_tokens):
        if token != "like":
            continue
        left = prefix_tokens[max(0, idx - 6):idx]
        if "recommend" in left:
            return True
    return False


def _context_reasons_for_visible_start(visible_tokens: list[str], visible_start: int) -> list[str]:
    reasons: list[str] = []
    for pattern, label in _TITLE_CONTEXT_PATTERNS:
        if visible_start >= len(pattern) and visible_tokens[visible_start - len(pattern):visible_start] == pattern:
            reasons.append(label)

    if visible_start >= 1 and visible_tokens[visible_start - 1] == "like":
        left = visible_tokens[max(0, visible_start - 6):visible_start]
        if "recommend" in left:
            reasons.append("recommend ... like")

    if visible_start >= 1 and visible_tokens[visible_start - 1] in {"and", "or"}:
        prefix = visible_tokens[:visible_start - 1]
        if _prefix_has_title_trigger(prefix):
            reasons.append(f'{visible_tokens[visible_start - 1]} after title context')

    return reasons


def _pos_token_payload(doc) -> list[dict]:
    tokens = []
    for i, token in enumerate(doc):
        tokens.append({
            "index": i,
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "tag": token.tag_,
            "dep": token.dep_,
            "shape": token.shape_,
            "is_alpha": bool(token.is_alpha),
            "is_stop": bool(token.is_stop),
            "is_punct": bool(token.is_punct),
            "like_num": bool(token.like_num),
        })
    return tokens


def _strip_det_tokens_from_span(original_tokens: list[str], original_doc, start: int, end: int) -> str:
    kept: list[str] = []
    for idx in range(start, end + 1):
        token = original_doc[idx]
        if token.is_punct:
            continue
        if token.pos_ == "DET":
            continue
        kept.append(original_tokens[idx])
    return _rebuild_query_from_tokens(kept)


def _run_phrase_stage(
    *,
    stage_name: str,
    original_doc,
    original_tokens: list[str],
    original_pos_payload: list[dict],
    consumed_mask: list[bool],
    max_anchor_span: int,
    canonical_phrase_map: dict[str, list[str]],
    canonical_title_map: dict[str, list[str]],
    resolver,
    placeholder: str,
    allowed_seed_pos: set[str] | None = None,
    require_context: bool = False,
    min_title_words: int = 1,
    every_word: bool = False,
) -> dict:
    visible_tokens, original_to_visible = _build_visible_tokens_with_mapping(original_tokens, consumed_mask, placeholder)
    working_query = _rebuild_query_from_tokens(visible_tokens)
    stage_info = {
        "stage_name": stage_name,
        "working_query": working_query,
        "anchor_attempts": [],
        "chosen_match": None,
    }

    for start, token in enumerate(original_doc):
        if consumed_mask[start] or token.is_punct:
            continue
        if not every_word:
            if allowed_seed_pos is None or token.pos_ not in allowed_seed_pos:
                continue

        anchor_attempt = {
            "stage_name": stage_name,
            "anchor_token_index": start,
            "anchor_token_text": token.text,
            "anchor_token_pos": token.pos_,
            "tried_phrases": [],
            "selected_match": None,
        }

        contiguous_end = start
        while contiguous_end < len(original_tokens) and not consumed_mask[contiguous_end] and not original_doc[contiguous_end].is_punct:
            contiguous_end += 1
        contiguous_count = contiguous_end - start
        max_len_here = min(max_anchor_span, contiguous_count)

        for span_len in range(max_len_here, 0, -1):
            raw_anchor = " ".join(original_tokens[start:start + span_len])
            canonical_anchor = canonicalize_title_text(raw_anchor)
            candidates = list(canonical_phrase_map.get(canonical_anchor, []))
            phrase_attempt = {
                "stage_name": stage_name,
                "span_len": span_len,
                "raw_anchor": raw_anchor,
                "canonical_anchor": canonical_anchor,
                "candidate_count": len(candidates),
                "candidate_titles": candidates[:20],
                "scored_candidates": [],
                "aligned_match": None,
            }

            if not candidates:
                anchor_attempt["tried_phrases"].append(phrase_attempt)
                continue

            visible_start = original_to_visible.get(start)
            context_reasons = _context_reasons_for_visible_start(visible_tokens, visible_start) if visible_start is not None else []
            sorted_candidates = sorted(candidates, key=lambda t: (-len(t.split()), -len(t), t))

            for candidate_title in sorted_candidates:
                alignment = _find_aligned_candidate_match(
                    candidate_title=candidate_title,
                    anchor_phrase=canonical_anchor,
                    anchor_start=start,
                    anchor_end=start + span_len - 1,
                    original_tokens=original_tokens,
                    consumed_mask=consumed_mask,
                )
                if alignment is None:
                    continue

                matched_token_count = len(candidate_title.split())
                accepted = matched_token_count >= min_title_words
                score = 1.0 if accepted else 0.0
                score_reasons: list[str] = []

                if require_context:
                    accepted = bool(context_reasons)
                    score = 1.0 if accepted else 0.0
                    score_reasons = list(context_reasons) if context_reasons else ["no title context"]
                else:
                    if accepted:
                        if min_title_words > 1:
                            score_reasons.append(f"exact aligned canonical match with {min_title_words}+ words")
                        else:
                            score_reasons.append("exact aligned canonical match")
                    else:
                        score_reasons.append(f"matched title has fewer than {min_title_words} words")

                qid_list = list(canonical_title_map.get(candidate_title, []))
                qid = qid_list[0] if qid_list else ""
                qid_short = qid if str(qid).startswith("Q") else str(qid).rsplit("/", 1)[-1]
                matched_title = _resolve_raw_title_for_canonical_qid(resolver, candidate_title, qid_short) if qid_short else candidate_title
                match_payload = {
                    "stage_name": stage_name,
                    "anchor_start_token": start,
                    "anchor_end_token": start + span_len - 1,
                    "anchor_raw_span": raw_anchor,
                    "anchor_canonical_span": canonical_anchor,
                    "message_start_token": alignment["message_start"],
                    "message_end_token": alignment["message_end"],
                    "candidate_anchor_start": alignment["candidate_anchor_start"],
                    "candidate_anchor_end": alignment["candidate_anchor_end"],
                    "matched_canonical_title": candidate_title,
                    "matched_title": matched_title,
                    "matched_token_count": matched_token_count,
                    "qid": qid_short,
                    "all_qids": [q if str(q).startswith("Q") else str(q).rsplit("/", 1)[-1] for q in qid_list],
                    "wikidata_url": f"https://www.wikidata.org/wiki/{qid_short}" if qid_short else None,
                    "score": score,
                    "score_reasons": score_reasons,
                    "visible_start_token": visible_start,
                }
                phrase_attempt["scored_candidates"].append(match_payload)

                if accepted:
                    phrase_attempt["aligned_match"] = match_payload
                    anchor_attempt["selected_match"] = match_payload
                    stage_info["chosen_match"] = match_payload
                    anchor_attempt["tried_phrases"].append(phrase_attempt)
                    stage_info["anchor_attempts"].append(anchor_attempt)
                    return stage_info

            anchor_attempt["tried_phrases"].append(phrase_attempt)

        stage_info["anchor_attempts"].append(anchor_attempt)

    return stage_info


def _run_det_stripped_stage(
    *,
    stage_name: str,
    original_doc,
    original_tokens: list[str],
    original_pos_payload: list[dict],
    consumed_mask: list[bool],
    max_anchor_span: int,
    det_stripped_title_map: dict[str, list[str]],
    canonical_title_map: dict[str, list[str]],
    resolver,
    placeholder: str,
    allowed_seed_pos: set[str],
) -> dict:
    visible_tokens, original_to_visible = _build_visible_tokens_with_mapping(original_tokens, consumed_mask, placeholder)
    working_query = _rebuild_query_from_tokens(visible_tokens)
    stage_info = {
        "stage_name": stage_name,
        "working_query": working_query,
        "anchor_attempts": [],
        "chosen_match": None,
    }

    for start, token in enumerate(original_doc):
        if consumed_mask[start] or token.is_punct or token.pos_ not in allowed_seed_pos:
            continue

        anchor_attempt = {
            "stage_name": stage_name,
            "anchor_token_index": start,
            "anchor_token_text": token.text,
            "anchor_token_pos": token.pos_,
            "tried_phrases": [],
            "selected_match": None,
        }

        contiguous_end = start
        while contiguous_end < len(original_tokens) and not consumed_mask[contiguous_end] and not original_doc[contiguous_end].is_punct:
            contiguous_end += 1
        contiguous_count = contiguous_end - start
        max_len_here = min(max_anchor_span, contiguous_count)

        for span_len in range(max_len_here, 0, -1):
            raw_anchor = " ".join(original_tokens[start:start + span_len])
            canonical_anchor = canonicalize_title_text(raw_anchor)
            det_stripped_anchor = _strip_det_tokens_from_span(original_tokens, original_doc, start, start + span_len - 1)
            det_stripped_anchor = strip_det_tokens_from_canonical_text(det_stripped_anchor)
            candidates = list(det_stripped_title_map.get(det_stripped_anchor, [])) if det_stripped_anchor else []
            phrase_attempt = {
                "stage_name": stage_name,
                "span_len": span_len,
                "raw_anchor": raw_anchor,
                "canonical_anchor": canonical_anchor,
                "det_stripped_anchor": det_stripped_anchor,
                "candidate_count": len(candidates),
                "candidate_titles": candidates[:20],
                "scored_candidates": [],
                "aligned_match": None,
            }

            if not candidates:
                anchor_attempt["tried_phrases"].append(phrase_attempt)
                continue

            visible_start = original_to_visible.get(start)
            stripped_token_count = len(det_stripped_anchor.split())
            sorted_candidates = sorted(candidates, key=lambda t: (-len(t.split()), -len(t), t))

            for candidate_title in sorted_candidates:
                qid_list = list(canonical_title_map.get(candidate_title, []))
                qid = qid_list[0] if qid_list else ""
                qid_short = qid if str(qid).startswith("Q") else str(qid).rsplit("/", 1)[-1]
                matched_title = _resolve_raw_title_for_canonical_qid(resolver, candidate_title, qid_short) if qid_short else candidate_title
                score = 1.0 if stripped_token_count >= 2 else 0.5
                score_reasons = [
                    "exact DET-stripped alias match" if stripped_token_count >= 2 else "exact DET-stripped alias match shorter than 2 words → flag 0.5",
                ]
                match_payload = {
                    "stage_name": stage_name,
                    "anchor_start_token": start,
                    "anchor_end_token": start + span_len - 1,
                    "anchor_raw_span": raw_anchor,
                    "anchor_canonical_span": canonical_anchor,
                    "det_stripped_anchor": det_stripped_anchor,
                    "message_start_token": start,
                    "message_end_token": start + span_len - 1,
                    "candidate_anchor_start": 0,
                    "candidate_anchor_end": stripped_token_count - 1,
                    "matched_canonical_title": candidate_title,
                    "matched_title": matched_title,
                    "matched_token_count": len(candidate_title.split()),
                    "alias_matched_token_count": stripped_token_count,
                    "qid": qid_short,
                    "all_qids": [q if str(q).startswith("Q") else str(q).rsplit("/", 1)[-1] for q in qid_list],
                    "wikidata_url": f"https://www.wikidata.org/wiki/{qid_short}" if qid_short else None,
                    "score": score,
                    "score_reasons": score_reasons,
                    "visible_start_token": visible_start,
                }
                phrase_attempt["scored_candidates"].append(match_payload)
                phrase_attempt["aligned_match"] = match_payload
                anchor_attempt["selected_match"] = match_payload
                stage_info["chosen_match"] = match_payload
                anchor_attempt["tried_phrases"].append(phrase_attempt)
                stage_info["anchor_attempts"].append(anchor_attempt)
                return stage_info

            anchor_attempt["tried_phrases"].append(phrase_attempt)

        stage_info["anchor_attempts"].append(anchor_attempt)

    return stage_info


def _run_fuzzy_det_stripped_stage(
    *,
    stage_name: str,
    original_doc,
    original_tokens: list[str],
    original_pos_payload: list[dict],
    consumed_mask: list[bool],
    max_anchor_span: int,
    det_stripped_title_map: dict[str, list[str]],
    canonical_title_map: dict[str, list[str]],
    resolver,
    placeholder: str,
    allowed_seed_pos: set[str],
    title_token_choices: list[str],
) -> dict:
    visible_tokens, original_to_visible = _build_visible_tokens_with_mapping(original_tokens, consumed_mask, placeholder)
    working_query = _rebuild_query_from_tokens(visible_tokens)
    stage_info = {
        "stage_name": stage_name,
        "working_query": working_query,
        "anchor_attempts": [],
        "chosen_match": None,
    }

    det_stripped_keys = list(det_stripped_title_map.keys())

    for start, token in enumerate(original_doc):
        if consumed_mask[start] or token.is_punct or token.pos_ not in allowed_seed_pos:
            continue

        anchor_attempt = {
            "stage_name": stage_name,
            "anchor_token_index": start,
            "anchor_token_text": token.text,
            "anchor_token_pos": token.pos_,
            "tried_phrases": [],
            "selected_match": None,
        }

        contiguous_end = start
        while contiguous_end < len(original_tokens) and not consumed_mask[contiguous_end] and not original_doc[contiguous_end].is_punct:
            contiguous_end += 1
        contiguous_count = contiguous_end - start
        max_len_here = min(max_anchor_span, contiguous_count)

        for span_len in range(max_len_here, 0, -1):
            raw_anchor = " ".join(original_tokens[start:start + span_len])
            canonical_anchor = canonicalize_title_text(raw_anchor)
            det_stripped_anchor = _strip_det_tokens_from_span(original_tokens, original_doc, start, start + span_len - 1)
            det_stripped_anchor = strip_det_tokens_from_canonical_text(det_stripped_anchor)
            typo_suspicions = _detect_typo_suspicions_in_text(det_stripped_anchor, title_token_choices) if det_stripped_anchor else []
            fuzzy_alias_hits = _extract_fuzzy_choices(det_stripped_anchor, det_stripped_keys, limit=8, score_cutoff=88.0) if typo_suspicions and det_stripped_anchor else []
            phrase_attempt = {
                "stage_name": stage_name,
                "span_len": span_len,
                "raw_anchor": raw_anchor,
                "canonical_anchor": canonical_anchor,
                "det_stripped_anchor": det_stripped_anchor,
                "typo_suspicions": typo_suspicions,
                "candidate_count": len(fuzzy_alias_hits),
                "candidate_titles": [choice for choice, _score in fuzzy_alias_hits[:20]],
                "fuzzy_candidates": [{"alias_key": choice, "fuzzy_score": round(float(score), 2)} for choice, score in fuzzy_alias_hits[:20]],
                "scored_candidates": [],
                "aligned_match": None,
            }

            if not typo_suspicions or not fuzzy_alias_hits:
                anchor_attempt["tried_phrases"].append(phrase_attempt)
                continue

            visible_start = original_to_visible.get(start)

            expanded_candidates: list[tuple[str, float]] = []
            for alias_key, fuzzy_score in fuzzy_alias_hits:
                for candidate_title in det_stripped_title_map.get(alias_key, []):
                    expanded_candidates.append((candidate_title, fuzzy_score))
            expanded_candidates.sort(key=lambda item: (-item[1], -len(item[0].split()), -len(item[0]), item[0]))

            for candidate_title, fuzzy_score in expanded_candidates:
                qid_list = list(canonical_title_map.get(candidate_title, []))
                qid = qid_list[0] if qid_list else ""
                qid_short = qid if str(qid).startswith("Q") else str(qid).rsplit("/", 1)[-1]
                matched_title = _resolve_raw_title_for_canonical_qid(resolver, candidate_title, qid_short) if qid_short else candidate_title
                score_reasons = [
                    f"fuzzy DET-stripped alias match (score {round(float(fuzzy_score), 2)})",
                    "typo signal triggered fuzzy fallback",
                ]
                if typo_suspicions:
                    typo_bits = ", ".join(f'{item["token"]}→{item["suggestion"]}' for item in typo_suspicions[:5])
                    score_reasons.append(f"typo-suspicious tokens: {typo_bits}")
                match_payload = {
                    "stage_name": stage_name,
                    "anchor_start_token": start,
                    "anchor_end_token": start + span_len - 1,
                    "anchor_raw_span": raw_anchor,
                    "anchor_canonical_span": canonical_anchor,
                    "det_stripped_anchor": det_stripped_anchor,
                    "message_start_token": start,
                    "message_end_token": start + span_len - 1,
                    "candidate_anchor_start": 0,
                    "candidate_anchor_end": max(0, len(det_stripped_anchor.split()) - 1),
                    "matched_canonical_title": candidate_title,
                    "matched_title": matched_title,
                    "matched_token_count": len(candidate_title.split()),
                    "alias_matched_token_count": len(det_stripped_anchor.split()),
                    "qid": qid_short,
                    "all_qids": [q if str(q).startswith("Q") else str(q).rsplit("/", 1)[-1] for q in qid_list],
                    "wikidata_url": f"https://www.wikidata.org/wiki/{qid_short}" if qid_short else None,
                    "score": 0.5,
                    "score_reasons": score_reasons,
                    "visible_start_token": visible_start,
                    "fuzzy_score": round(float(fuzzy_score), 2),
                    "typo_suspicions": typo_suspicions,
                }
                phrase_attempt["scored_candidates"].append(match_payload)
                phrase_attempt["aligned_match"] = match_payload
                anchor_attempt["selected_match"] = match_payload
                stage_info["chosen_match"] = match_payload
                anchor_attempt["tried_phrases"].append(phrase_attempt)
                stage_info["anchor_attempts"].append(anchor_attempt)
                return stage_info

            anchor_attempt["tried_phrases"].append(phrase_attempt)

        stage_info["anchor_attempts"].append(anchor_attempt)

    return stage_info


def _debug_pos_driven_title_extraction(query: str, resolver) -> dict:
    placeholder = "plazeholdaqqq"
    max_anchor_span = 5
    canonical_title_map = _load_canonical_title_map(resolver)
    canonical_phrase_map = _load_canonical_phrase_map(resolver, max_span_len=max_anchor_span)
    det_stripped_title_map = _load_det_stripped_title_map(resolver)
    title_token_choices = _build_title_token_choices(canonical_title_map, det_stripped_title_map)
    canonical_input = canonicalize_title_text(query)

    pos = POS()
    original_doc = pos.doc(canonical_input)
    original_tokens = [token.text for token in original_doc]
    original_pos_payload = _pos_token_payload(original_doc)
    consumed_mask = [False] * len(original_tokens)

    extracted_items: list[dict] = []
    passes: list[dict] = []

    stage_configs = [
        {
            "stage_name": "Stage 1 · NOUN/PROPN/DET exact",
            "runner": "phrase",
            "allowed_seed_pos": {"NOUN", "PROPN", "DET"},
            "every_word": False,
            "require_context": False,
            "min_title_words": 1,
        },
        {
            "stage_name": "Stage 2 · pattern recognition",
            "runner": "phrase",
            "allowed_seed_pos": None,
            "every_word": True,
            "require_context": True,
            "min_title_words": 1,
        },
        {
            "stage_name": "Stage 3 · broad exact 3+ words",
            "runner": "phrase",
            "allowed_seed_pos": None,
            "every_word": True,
            "require_context": False,
            "min_title_words": 3,
        },
        {
            "stage_name": "Stage 4 · DET-stripped alias fallback",
            "runner": "det_stripped",
            "allowed_seed_pos": {"NOUN", "PROPN", "DET"},
            "every_word": False,
        },
        {
            "stage_name": "Stage 5 · typo-triggered fuzzy fallback",
            "runner": "fuzzy_det_stripped",
            "allowed_seed_pos": {"NOUN", "PROPN", "DET"},
            "every_word": False,
        },
    ]

    while True:
        visible_tokens, _ = _build_visible_tokens_with_mapping(original_tokens, consumed_mask, placeholder)
        working_query = _rebuild_query_from_tokens(visible_tokens)
        pass_info = {
            "pass_index": len(passes) + 1,
            "working_query": working_query,
            "tokens": original_pos_payload,
            "stage_order": [cfg["stage_name"] for cfg in stage_configs],
            "anchor_attempts": [],
            "stages": [],
            "match": None,
            "replacement_query": None,
        }

        chosen = None
        for cfg in stage_configs:
            if cfg["runner"] == "phrase":
                stage_result = _run_phrase_stage(
                    stage_name=cfg["stage_name"],
                    original_doc=original_doc,
                    original_tokens=original_tokens,
                    original_pos_payload=original_pos_payload,
                    consumed_mask=consumed_mask,
                    max_anchor_span=max_anchor_span,
                    canonical_phrase_map=canonical_phrase_map,
                    canonical_title_map=canonical_title_map,
                    resolver=resolver,
                    placeholder=placeholder,
                    allowed_seed_pos=cfg.get("allowed_seed_pos"),
                    require_context=cfg.get("require_context", False),
                    min_title_words=cfg.get("min_title_words", 1),
                    every_word=cfg.get("every_word", False),
                )
            elif cfg["runner"] == "det_stripped":
                stage_result = _run_det_stripped_stage(
                    stage_name=cfg["stage_name"],
                    original_doc=original_doc,
                    original_tokens=original_tokens,
                    original_pos_payload=original_pos_payload,
                    consumed_mask=consumed_mask,
                    max_anchor_span=max_anchor_span,
                    det_stripped_title_map=det_stripped_title_map,
                    canonical_title_map=canonical_title_map,
                    resolver=resolver,
                    placeholder=placeholder,
                    allowed_seed_pos=cfg["allowed_seed_pos"],
                )
            else:
                stage_result = _run_fuzzy_det_stripped_stage(
                    stage_name=cfg["stage_name"],
                    original_doc=original_doc,
                    original_tokens=original_tokens,
                    original_pos_payload=original_pos_payload,
                    consumed_mask=consumed_mask,
                    max_anchor_span=max_anchor_span,
                    det_stripped_title_map=det_stripped_title_map,
                    canonical_title_map=canonical_title_map,
                    resolver=resolver,
                    placeholder=placeholder,
                    allowed_seed_pos=cfg["allowed_seed_pos"],
                    title_token_choices=title_token_choices,
                )

            pass_info["stages"].append(stage_result)
            pass_info["anchor_attempts"].extend(stage_result.get("anchor_attempts", []))
            if stage_result.get("chosen_match") is not None:
                chosen = stage_result["chosen_match"]
                break

        if chosen is None:
            passes.append(pass_info)
            break

        pass_info["match"] = chosen
        extracted_items.append({
            "title": chosen["matched_title"],
            "qid": chosen["qid"],
            "wikidata_url": chosen["wikidata_url"],
            "score": chosen["score"],
        })

        for idx in range(chosen["message_start_token"], chosen["message_end_token"] + 1):
            consumed_mask[idx] = True

        replacement_query = _build_placeholder_view_from_consumed(original_tokens, consumed_mask, placeholder)
        pass_info["replacement_query"] = replacement_query
        passes.append(pass_info)

    final_query = _build_placeholder_view_from_consumed(original_tokens, consumed_mask, placeholder)

    return {
        "original_query": query,
        "canonical_input_query": canonical_input,
        "placeholder": placeholder,
        "max_anchor_span": max_anchor_span,
        "stage_order": [cfg["stage_name"] for cfg in stage_configs],
        "context_patterns": [label for _pattern, label in _TITLE_CONTEXT_PATTERNS] + ["recommend ... like", "and/or after title context"],
        "extracted_items": extracted_items,
        "final_query": final_query,
        "passes": passes,
        "title_count": len(extracted_items),
        "canonical_dictionary_title_count": len(canonical_title_map),
        "canonical_phrase_dictionary_count": len(canonical_phrase_map),
        "det_stripped_dictionary_count": len(det_stripped_title_map),
        "title_token_choice_count": len(title_token_choices),
        "fuzzy_backend": "rapidfuzz" if process is not None and fuzz is not None else "difflib",
    }


def _title_payload(title: str, qid: str) -> dict:
    qid = str(qid)
    if not qid.startswith("Q"):
        qid = qid.rsplit("/", 1)[-1]
    return {
        "title": title,
        "qid": qid,
        "wikidata_url": f"https://www.wikidata.org/wiki/{qid}",
    }


def _normalize_qid_lookup_text(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    raw = raw.rsplit("/", 1)[-1].strip()
    raw = raw.upper()
    if raw.startswith("Q"):
        raw = "Q" + raw[1:].strip()
    if raw.isdigit():
        raw = f"Q{raw}"
    return raw


def _invert_title_to_qids_map(title_to_qids_map: dict[str, list[str]]) -> dict[str, list[str]]:
    qid_to_titles: dict[str, list[str]] = {}
    for title, qids in title_to_qids_map.items():
        for qid in qids:
            qid_to_titles.setdefault(qid, []).append(title)
    for qid, titles in qid_to_titles.items():
        qid_to_titles[qid] = sorted(dict.fromkeys(titles), key=lambda t: (len(t), t))
    return qid_to_titles




def _iri_to_qid_short(value: str | None) -> str | None:
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.startswith("Q") and s[1:].isdigit():
        return s
    tail = s.rsplit("/", 1)[-1]
    if re.fullmatch(r"Q\d+", tail):
        return tail
    return None


def _wikidata_url_for(value: str | None) -> str | None:
    qid = _iri_to_qid_short(value)
    return f"https://www.wikidata.org/wiki/{qid}" if qid else None


def _normalize_entity_label_lookup(text: str) -> str:
    return _normalize_lookup_text(text)


def _collect_same_name_candidates(agent: Agent, query: str) -> list[dict]:
    normalized = _normalize_entity_label_lookup(query)
    candidates: list[dict] = []
    seen: set[tuple[str, str]] = set()

    title_to_qids_map, _meta = _load_title_to_qids_map(agent.title_resolver)
    movie_qids = title_to_qids_map.get(normalized, [])
    for qid in movie_qids:
        iri = f"http://www.wikidata.org/entity/{qid}"
        title = agent.id_movie_dict.get(iri, normalized)
        key = ("movie", iri)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({
            "kind": "movie",
            "id": iri,
            "qid": qid,
            "label": title,
            "display_name": title,
            "subtitle": f"Movie title · {qid}",
            "wikidata_url": _wikidata_url_for(qid),
        })

    for iri, label in agent.id_movie_dict.items():
        if _normalize_entity_label_lookup(label) != normalized:
            continue
        qid = _iri_to_qid_short(iri)
        key = ("movie", iri)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({
            "kind": "movie",
            "id": iri,
            "qid": qid,
            "label": label,
            "display_name": label,
            "subtitle": f"Movie title · {qid or iri}",
            "wikidata_url": _wikidata_url_for(iri),
        })

    for iri, label in agent.id_entity_dict.items():
        if _normalize_entity_label_lookup(label) != normalized:
            continue
        qid = _iri_to_qid_short(iri)
        key = ("entity", iri)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({
            "kind": "entity",
            "id": iri,
            "qid": qid,
            "label": label,
            "display_name": label,
            "subtitle": f"Entity label · {qid or iri}",
            "wikidata_url": _wikidata_url_for(iri),
        })

    candidates.sort(key=lambda item: (item["kind"] != "movie", item["display_name"].lower(), item["qid"] or item["id"]))
    return candidates



def _label_value_payload(agent: Agent, value: str) -> dict:
    if value in agent.id_movie_dict:
        label = agent.id_movie_dict[value]
        qid = _iri_to_qid_short(value)
        return {
            "kind": "movie",
            "id": value,
            "qid": qid,
            "label": label,
            "display_name": label,
            "wikidata_url": _wikidata_url_for(value),
            "raw": value,
        }
    if value in agent.id_entity_dict:
        label = agent.id_entity_dict[value]
        qid = _iri_to_qid_short(value)
        return {
            "kind": "entity",
            "id": value,
            "qid": qid,
            "label": label,
            "display_name": label,
            "wikidata_url": _wikidata_url_for(value),
            "raw": value,
        }
    if value in agent.id_relation_dict:
        label = agent.id_relation_dict[value]
        qid = _iri_to_qid_short(value)
        return {
            "kind": "relation",
            "id": value,
            "qid": qid,
            "label": label,
            "display_name": label,
            "wikidata_url": _wikidata_url_for(value),
            "raw": value,
        }
    qid = _iri_to_qid_short(value)
    return {
        "kind": "literal",
        "id": value,
        "qid": qid,
        "label": str(value),
        "display_name": str(value),
        "wikidata_url": _wikidata_url_for(value),
        "raw": value,
    }



def _build_movie_detail_payload(agent: Agent, movie_iri: str) -> dict:
    label = agent.id_movie_dict.get(movie_iri, movie_iri)
    qid = _iri_to_qid_short(movie_iri)
    attrs = agent.movie_attrs.get(movie_iri, {}) or {}
    relation_groups: list[dict] = []

    for relation_iri in sorted(attrs.keys(), key=lambda r: agent.id_relation_dict.get(r, r)):
        relation_label = agent.id_relation_dict.get(relation_iri, relation_iri)
        values = sorted(attrs.get(relation_iri, set()), key=lambda v: (_label_value_payload(agent, v)["display_name"].lower(), str(v)))
        value_payloads = [_label_value_payload(agent, v) for v in values]
        relation_groups.append({
            "relation_iri": relation_iri,
            "relation_qid": _iri_to_qid_short(relation_iri),
            "relation_label": relation_label,
            "relation_wikidata_url": _wikidata_url_for(relation_iri),
            "value_count": len(value_payloads),
            "values": value_payloads,
        })

    return {
        "selection_kind": "movie",
        "id": movie_iri,
        "qid": qid,
        "label": label,
        "wikidata_url": _wikidata_url_for(movie_iri),
        "relation_group_count": len(relation_groups),
        "relation_groups": relation_groups,
        "summary": f"Movie '{label}' has {len(relation_groups)} outgoing relation types in movie_attrs.",
    }



def _build_entity_detail_payload(agent: Agent, entity_iri: str) -> dict:
    label = agent.id_entity_dict.get(entity_iri, entity_iri)
    qid = _iri_to_qid_short(entity_iri)
    movie_groups: list[dict] = []
    total_movies = 0

    for relation_iri in sorted(agent.attr_index.keys(), key=lambda r: agent.id_relation_dict.get(r, r)):
        entity_to_movies = agent.attr_index.get(relation_iri, {}) or {}
        if entity_iri not in entity_to_movies:
            continue
        movie_iris = sorted(entity_to_movies.get(entity_iri, set()), key=lambda m: agent.id_movie_dict.get(m, m).lower())
        movie_payloads = []
        for movie_iri in movie_iris:
            movie_label = agent.id_movie_dict.get(movie_iri, movie_iri)
            movie_qid = _iri_to_qid_short(movie_iri)
            movie_payloads.append({
                "kind": "movie",
                "id": movie_iri,
                "qid": movie_qid,
                "label": movie_label,
                "display_name": movie_label,
                "wikidata_url": _wikidata_url_for(movie_iri),
            })
        total_movies += len(movie_payloads)
        movie_groups.append({
            "relation_iri": relation_iri,
            "relation_qid": _iri_to_qid_short(relation_iri),
            "relation_label": agent.id_relation_dict.get(relation_iri, relation_iri),
            "relation_wikidata_url": _wikidata_url_for(relation_iri),
            "movie_count": len(movie_payloads),
            "movies": movie_payloads,
        })

    return {
        "selection_kind": "entity",
        "id": entity_iri,
        "qid": qid,
        "label": label,
        "wikidata_url": _wikidata_url_for(entity_iri),
        "relation_group_count": len(movie_groups),
        "total_movies": total_movies,
        "movie_groups": movie_groups,
        "summary": f"Entity '{label}' is linked from {total_movies} movie entries across {len(movie_groups)} relation types in attr_index.",
    }


def _bag_value_label(agent: Agent, raw_value: str) -> str:
    if raw_value in agent.id_movie_dict:
        return agent.id_movie_dict[raw_value]
    if raw_value in agent.id_entity_dict:
        return agent.id_entity_dict[raw_value]
    if raw_value in agent.id_relation_dict:
        return agent.id_relation_dict[raw_value]
    qid = _iri_to_qid_short(raw_value)
    if qid:
        iri = f"http://www.wikidata.org/entity/{qid}"
        if iri in agent.id_movie_dict:
            return agent.id_movie_dict[iri]
        if iri in agent.id_entity_dict:
            return agent.id_entity_dict[iri]
    return str(raw_value)



def _profile_rows_payload(agent: Agent, preference_profile: dict[tuple[str, str], float]) -> list[dict]:
    rows = []
    for (group, value), score in preference_profile.items():
        if float(score) <= 0:
            continue
        rows.append({
            "group": group,
            "relation_iri": group,
            "relation_qid": _iri_to_qid_short(group),
            "relation_label": agent.id_relation_dict.get(group, group),
            "value": value,
            "raw_value": value,
            "value_qid": _iri_to_qid_short(value),
            "value_label": _bag_value_label(agent, value),
            "score": float(score),
        })
    rows.sort(key=lambda item: (-item["score"], item["relation_label"], item["value_label"]))
    return rows



def _recommendation_bag_report_payload(agent: Agent, query: str, limit: int = 5) -> dict:
    extracted = agent.extractor.extract_rec_mulm(query)
    if extracted is None:
        titles, entities, clarify_titles = [], [], []
    else:
        titles = list(extracted.get("titles", []))
        entities = list(extracted.get("entities", []))
        clarify_titles = list(extracted.get("clarify_titles", []))
    liked_qids_full = agent.title_resolver.qid_from_title(titles)
    attr_qids = agent._recommendation_attr_ids_from_entities(entities)

    preference_profile, content_ranked = agent.conrec.rank_movies_by_profile(liked_qids_full, attr_qids)
    profile_rows = _profile_rows_payload(agent, preference_profile)
    content_rank_lookup = {movie_id: idx + 1 for idx, (movie_id, _score) in enumerate(content_ranked)}
    content_score_lookup = {movie_id: float(score) for movie_id, score in content_ranked}

    final_rec_ids = agent.conrec.recommend(liked_qids_full, attr_qids, limit=limit) if (liked_qids_full or attr_qids) else []
    recommendation_source = "content_or_knn"
    if not final_rec_ids and attr_qids:
        final_rec_ids = agent.conrec.recommend_for_entities(attr_qids, limit=limit)
        recommendation_source = "entity_fallback"
    recommendations = []
    for idx, movie_id in enumerate(final_rec_ids, start=1):
        movie_title = agent.id_movie_dict.get(movie_id) or movie_id
        movie_qid = _iri_to_qid_short(movie_id)
        explanation = agent.conrec.explain_movie_against_profile(movie_id, preference_profile)
        matched_rows = []
        for row in explanation.get("matched_attributes", []):
            group = str(row.get("group"))
            value = str(row.get("value"))
            score = float(row.get("score", 0.0))
            if score <= 0:
                continue
            matched_rows.append({
                "group": group,
                "relation_iri": group,
                "relation_qid": _iri_to_qid_short(group),
                "relation_label": agent.id_relation_dict.get(group, group),
                "value": value,
                "raw_value": value,
                "value_qid": _iri_to_qid_short(value),
                "value_label": _bag_value_label(agent, value),
                "score": score,
            })
        matched_rows.sort(key=lambda item: (-item["score"], item["relation_label"], item["value_label"]))
        recommendations.append({
            "final_rank": idx,
            "movie_id": movie_id,
            "movie_qid": movie_qid,
            "title": movie_title,
            "content_rank": content_rank_lookup.get(movie_id),
            "content_score": float(content_score_lookup.get(movie_id, explanation.get("total_score", 0.0))),
            "match_count": len(matched_rows),
            "matched_attributes": matched_rows,
        })

    return {
        "query": query,
        "extracted_titles": titles,
        "liked_qids": liked_qids_full,
        "extracted_entities": entities,
        "entity_qids": attr_qids,
        "preference_profile": profile_rows,
        "preference_profile_count": len(profile_rows),
        "recommendations": recommendations,
        "recommendation_count": len(recommendations),
        "recommendation_source": recommendation_source,
    }


def _recommendation_attribute_overlap_report_payload(agent: Agent, query: str) -> dict:
    extracted = agent.extractor.extract_rec_mulm(query)
    if extracted is None:
        titles, entities = [], []
    else:
        titles = list(extracted.get("titles", []))
        entities = list(extracted.get("entities", []))

    liked_qids_full = agent.title_resolver.qid_from_title(titles)
    debug_rows = agent.conrec.debug_liked_movie_profile_rows(liked_qids_full)

    liked_movie_payloads = []
    movie_title_lookup = {}
    for movie_id in liked_qids_full:
        title = agent.id_movie_dict.get(movie_id, movie_id)
        movie_title_lookup[movie_id] = title
        liked_movie_payloads.append({
            "movie_id": movie_id,
            "movie_qid": _iri_to_qid_short(movie_id),
            "title": title,
            "wikidata_url": _wikidata_url_for(movie_id),
        })

    rows = []
    for row in debug_rows:
        source_movies = []
        for movie_id in row.get("source_movie_ids", []):
            source_movies.append({
                "movie_id": movie_id,
                "movie_qid": _iri_to_qid_short(movie_id),
                "title": movie_title_lookup.get(movie_id, agent.id_movie_dict.get(movie_id, movie_id)),
            })
        rows.append({
            "relation_iri": row.get("group"),
            "relation_qid": _iri_to_qid_short(str(row.get("group") or "")),
            "relation_label": agent.id_relation_dict.get(str(row.get("group") or ""), str(row.get("group") or "")),
            "value": row.get("value"),
            "value_qid": _iri_to_qid_short(str(row.get("value") or "")),
            "value_label": _bag_value_label(agent, str(row.get("value") or "")),
            "score": float(row.get("score", 0.0)),
            "base_weight": float(row.get("base_weight", 0.0)),
            "multiplier": float(row.get("multiplier", 0.0)),
            "multiplier_reason": str(row.get("multiplier_reason") or ""),
            "overlap_count": int(row.get("overlap_count", 0)),
            "source_movies": source_movies,
        })

    return {
        "query": query,
        "extracted_titles": titles,
        "liked_movies": liked_movie_payloads,
        "liked_movie_count": len(liked_movie_payloads),
        "extracted_entities": entities,
        "rows": rows,
        "row_count": len(rows),
    }



@app.get("/api/health")
def health():
    agent = get_agent()
    if agent is None:
        return jsonify({
            "ok": False,
            "agent_loaded": False,
            "dataset_dir": str(DEFAULT_DATASET_DIR),
            "artifacts_dir": str(DEFAULT_ARTIFACTS_DIR),
            "missing_artifacts": RuntimeArtifactStore.validate(DEFAULT_ARTIFACTS_DIR),
            "error": _agent_error,
        }), 503
    return jsonify({
        "ok": True,
        "agent_loaded": True,
        "dataset_dir": str(DEFAULT_DATASET_DIR),
        "artifacts_dir": str(DEFAULT_ARTIFACTS_DIR),
    })





@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"ok": False, "error": "Message is required."}), 400

    agent = get_agent()
    if agent is None:
        return jsonify({
            "ok": False,
            "error": (
                "The movie agent could not be initialized. "
                "Check DATAPATH and confirm runtime_artifacts/, titles_to_qid.json, and ratings/ are present."
            ),
            "details": _agent_error,
            "dataset_dir": str(DEFAULT_DATASET_DIR),
            "artifacts_dir": str(DEFAULT_ARTIFACTS_DIR),
            "missing_artifacts": RuntimeArtifactStore.validate(DEFAULT_ARTIFACTS_DIR),
        }), 503

    try:
        response_text = agent.handle_message(message)
        return jsonify({"ok": True, "response": response_text})
    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
