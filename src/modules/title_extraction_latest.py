from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from modules.nlp.pos import POS
from modules.paths import get_dataset_dir
from modules.title_canonicalizer import (
    build_canonical_phrase_to_titles,
    build_canonical_title_to_qids,
    build_det_stripped_title_to_titles,
    canonical_phrase_dict_path,
    canonical_title_dict_path,
    canonicalize_title_text,
    det_stripped_title_dict_path,
    load_canonical_phrase_dict,
    load_canonical_title_dict,
    load_det_stripped_title_dict,
    strip_det_tokens_from_canonical_text,
)

try:
    from rapidfuzz import fuzz, process
except Exception:  # pragma: no cover
    fuzz = None
    process = None

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


def _load_canonical_title_map(resolver, dataset_dir) -> dict[str, list[str]]:
    path = canonical_title_dict_path(dataset_dir)
    if path.exists():
        try:
            return load_canonical_title_dict(dataset_dir)
        except Exception:
            pass
    return build_canonical_title_to_qids(resolver.t_q_dict)


def _load_canonical_phrase_map(resolver, dataset_dir, max_span_len: int = 5) -> dict[str, list[str]]:
    path = canonical_phrase_dict_path(dataset_dir, max_span_len=max_span_len)
    if path.exists():
        try:
            return load_canonical_phrase_dict(dataset_dir, max_span_len=max_span_len)
        except Exception:
            pass
    canonical_title_map = _load_canonical_title_map(resolver, dataset_dir)
    return build_canonical_phrase_to_titles(canonical_title_map, max_span_len=max_span_len)


def _load_det_stripped_title_map(resolver, dataset_dir) -> dict[str, list[str]]:
    path = det_stripped_title_dict_path(dataset_dir)
    if path.exists():
        try:
            return load_det_stripped_title_dict(dataset_dir)
        except Exception:
            pass
    canonical_title_map = _load_canonical_title_map(resolver, dataset_dir)
    return build_det_stripped_title_to_titles(canonical_title_map)


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


def _detect_typo_suspicions_in_text(text: str, title_token_choices: list[str]) -> list[dict[str, Any]]:
    suspicions: list[dict[str, Any]] = []
    for token in canonicalize_title_text(text).split():
        if not token or token.isdigit() or len(token) <= 2:
            continue
        if token in title_token_choices:
            continue
        suggestions = _extract_fuzzy_choices(token, title_token_choices, limit=1, score_cutoff=80.0)
        if not suggestions:
            continue
        suggestion, score = suggestions[0]
        suspicions.append({"token": token, "suggestion": suggestion, "score": round(float(score), 2)})
    return suspicions


def _resolve_raw_title_for_canonical_qid(resolver, canonical_title: str, qid_short: str) -> str:
    for raw_title, raw_qid in resolver.t_q_dict.items():
        raw_qid_short = raw_qid if str(raw_qid).startswith("Q") else str(raw_qid).rsplit("/", 1)[-1]
        if raw_qid_short == qid_short and canonicalize_title_text(raw_title) == canonical_title:
            return raw_title
    return canonical_title


def _normalize_lookup_text(text: str) -> str:
    s = (text or "").strip().lower()
    s = s.replace("’", "'").replace("‘", "'").replace('“', '"').replace('”', '"')
    s = re.sub(r"\s+", " ", s)
    return s


def _normalize_surface_text_preserve_punct(text: str) -> str:
    s = (text or "").strip().lower()
    s = s.replace("’", "'").replace("‘", "'").replace('“', '"').replace('”', '"')
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace("…", "...")
    s = re.sub(r"\s+", " ", s)
    return s


def _surface_token_text(token) -> str:
    normalized = _normalize_surface_text_preserve_punct(token.text)
    return normalized if normalized else token.text.lower()


_SENTENCE_STOP_TOKENS = {".", "!", "?"}
_TITLE_LIST_JOINER_TOKENS = {",", "and", "or"}


def _sentence_start_index(tokens: list[str], end_exclusive: int) -> int:
    for idx in range(end_exclusive - 1, -1, -1):
        if tokens[idx] in _SENTENCE_STOP_TOKENS:
            return idx + 1
    return 0


def _find_title_trigger_spans(tokens: list[str], end_exclusive: int) -> list[tuple[int, int, str]]:
    sentence_start = _sentence_start_index(tokens, end_exclusive)
    spans: list[tuple[int, int, str]] = []
    for pattern, label in _TITLE_CONTEXT_PATTERNS:
        if end_exclusive - sentence_start < len(pattern):
            continue
        for start in range(sentence_start, end_exclusive - len(pattern) + 1):
            if tokens[start:start + len(pattern)] == pattern:
                spans.append((start, start + len(pattern), label))
    for idx in range(sentence_start, end_exclusive):
        if tokens[idx] != "like":
            continue
        left = tokens[max(sentence_start, idx - 6):idx]
        if "recommend" in left:
            spans.append((idx, idx + 1, "recommend ... like"))
    return spans


def _gap_looks_like_title_list_continuation(gap_tokens: list[str], placeholder: str) -> bool:
    if not gap_tokens:
        return False
    allowed = set(_TITLE_LIST_JOINER_TOKENS) | {placeholder}
    return all(token in allowed for token in gap_tokens) and placeholder in gap_tokens


def _non_det_word_count(text: str) -> int:
    stripped = strip_det_tokens_from_canonical_text(text)
    return len(stripped.split()) if stripped else 0


def _tokens_for_later_disambiguation(*, raw_anchor: str, det_stripped_anchor: str | None, candidate_title: str) -> list[str]:
    primary = det_stripped_anchor or strip_det_tokens_from_canonical_text(raw_anchor) or strip_det_tokens_from_canonical_text(candidate_title)
    return [tok for tok in primary.split() if tok]


def _deferred_match_key(match_payload: dict[str, Any]) -> tuple[Any, ...]:
    return (
        match_payload.get("matched_title"),
        match_payload.get("message_start_token"),
        match_payload.get("message_end_token"),
    )


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


def _context_reasons_for_visible_start(visible_tokens: list[str], visible_start: int, placeholder: str) -> list[str]:
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

    trigger_spans = _find_title_trigger_spans(visible_tokens, visible_start)
    for _trigger_start, trigger_end, label in trigger_spans:
        gap_tokens = visible_tokens[trigger_end:visible_start]
        if _gap_looks_like_title_list_continuation(gap_tokens, placeholder):
            if "," in gap_tokens:
                reasons.append(f'comma-separated title list after {label}')
            else:
                reasons.append(f'title list continuation after {label}')

    deduped: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            deduped.append(reason)
    return deduped


def _pos_token_payload(doc) -> list[dict[str, Any]]:
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
        if token.is_punct or token.pos_ == "DET":
            continue
        kept.append(original_tokens[idx])
    return _rebuild_query_from_tokens(kept)


def _find_aligned_candidate_match(candidate_title: str, anchor_phrase: str, anchor_start: int, anchor_end: int, original_tokens: list[str], consumed_mask: list[bool]) -> dict[str, Any] | None:
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
        if any(consumed_mask[idx] for idx in range(message_start, message_end + 1)):
            continue
        message_tokens = original_tokens[message_start:message_end + 1]
        if message_tokens != candidate_tokens:
            continue
        return {
            "message_start_token": message_start,
            "message_end_token": message_end,
            "candidate_anchor_start": cand_anchor_start,
            "candidate_anchor_end": cand_anchor_start + anchor_len - 1,
        }
    return None


def _run_phrase_stage(*, stage_name: str, original_doc, original_tokens: list[str], original_pos_payload: list[dict[str, Any]], consumed_mask: list[bool], max_anchor_span: int, canonical_phrase_map: dict[str, list[str]], canonical_title_map: dict[str, list[str]], resolver, placeholder: str, allowed_seed_pos: set[str] | None = None, require_context: bool = False, min_title_words: int = 1, every_word: bool = False) -> dict[str, Any]:
    visible_tokens, original_to_visible = _build_visible_tokens_with_mapping(original_tokens, consumed_mask, placeholder)
    working_query = _rebuild_query_from_tokens(visible_tokens)
    stage_info: dict[str, Any] = {"stage_name": stage_name, "working_query": working_query, "anchor_attempts": [], "chosen_match": None, "deferred_matches": []}
    for start, token in enumerate(original_doc):
        if consumed_mask[start] or token.is_punct:
            continue
        if not every_word and (allowed_seed_pos is None or token.pos_ not in allowed_seed_pos):
            continue
        anchor_attempt: dict[str, Any] = {"stage_name": stage_name, "anchor_token_index": start, "anchor_token_text": token.text, "anchor_token_pos": token.pos_, "tried_phrases": [], "selected_match": None}
        contiguous_end = start
        while contiguous_end < len(original_tokens) and not consumed_mask[contiguous_end] and not original_doc[contiguous_end].is_punct:
            contiguous_end += 1
        contiguous_count = contiguous_end - start
        max_len_here = min(max_anchor_span, contiguous_count)
        for span_len in range(max_len_here, 0, -1):
            raw_anchor = " ".join(original_tokens[start:start + span_len])
            canonical_anchor = canonicalize_title_text(raw_anchor)
            candidates = list(canonical_phrase_map.get(canonical_anchor, []))
            phrase_attempt: dict[str, Any] = {"stage_name": stage_name, "span_len": span_len, "raw_anchor": raw_anchor, "canonical_anchor": canonical_anchor, "candidate_count": len(candidates), "candidate_titles": candidates[:20], "scored_candidates": [], "aligned_match": None}
            if not candidates:
                anchor_attempt["tried_phrases"].append(phrase_attempt)
                continue
            visible_start = original_to_visible.get(start)
            context_reasons = _context_reasons_for_visible_start(visible_tokens, visible_start, placeholder) if visible_start is not None else []
            sorted_candidates = sorted(candidates, key=lambda t: (-len(t.split()), -len(t), t))
            for candidate_title in sorted_candidates:
                alignment = _find_aligned_candidate_match(candidate_title, canonical_anchor, start, start + span_len - 1, original_tokens, consumed_mask)
                if alignment is None:
                    continue
                matched_token_count = len(candidate_title.split())
                non_det_word_count = _non_det_word_count(candidate_title)
                short_match_needs_context = min_title_words <= 1 and non_det_word_count < 2
                accepted = matched_token_count >= min_title_words
                score = 1.0 if accepted else 0.0
                score_reasons: list[str] = []
                deferred_reason: str | None = None
                if require_context:
                    accepted = bool(context_reasons)
                    score = 1.0 if accepted else 0.0
                    score_reasons = list(context_reasons) if context_reasons else ["no title context"]
                else:
                    if accepted and short_match_needs_context:
                        accepted = bool(context_reasons)
                        if accepted:
                            score_reasons = ["short exact aligned canonical match rescued by title context", *context_reasons]
                        else:
                            score = 0.5
                            score_reasons = ["short exact aligned canonical match without title context → defer"]
                            deferred_reason = "short exact aligned canonical match without title context"
                    elif accepted:
                        score_reasons.append("exact aligned canonical match" if min_title_words <= 1 else f"exact aligned canonical match with {min_title_words}+ words")
                    else:
                        score_reasons.append(f"exact aligned canonical match but title has fewer than {min_title_words} words")
                qid_list = canonical_title_map.get(candidate_title, [])
                qid = qid_list[0] if qid_list else ""
                qid_short = qid if str(qid).startswith("Q") else str(qid).rsplit("/", 1)[-1]
                matched_title = _resolve_raw_title_for_canonical_qid(resolver, candidate_title, qid_short) if qid_short else candidate_title
                match_payload = {
                    "stage_name": stage_name,
                    "anchor_start_token": start,
                    "anchor_end_token": start + span_len - 1,
                    "anchor_raw_span": raw_anchor,
                    "anchor_canonical_span": canonical_anchor,
                    "message_start_token": alignment["message_start_token"],
                    "message_end_token": alignment["message_end_token"],
                    "candidate_anchor_start": alignment["candidate_anchor_start"],
                    "candidate_anchor_end": alignment["candidate_anchor_end"],
                    "matched_canonical_title": candidate_title,
                    "matched_title": matched_title,
                    "matched_token_count": matched_token_count,
                    "non_det_word_count": non_det_word_count,
                    "match_tokens_no_det": _tokens_for_later_disambiguation(raw_anchor=raw_anchor, det_stripped_anchor=None, candidate_title=candidate_title),
                    "qid": qid_short,
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
                if deferred_reason:
                    deferred_payload = dict(match_payload)
                    deferred_payload["defer_reason"] = deferred_reason
                    stage_info["deferred_matches"].append(deferred_payload)
            anchor_attempt["tried_phrases"].append(phrase_attempt)
        stage_info["anchor_attempts"].append(anchor_attempt)
    return stage_info


def _run_det_stripped_stage(*, stage_name: str, original_doc, original_tokens: list[str], original_pos_payload: list[dict[str, Any]], consumed_mask: list[bool], max_anchor_span: int, det_stripped_title_map: dict[str, list[str]], canonical_title_map: dict[str, list[str]], resolver, placeholder: str, allowed_seed_pos: set[str]) -> dict[str, Any]:
    visible_tokens, original_to_visible = _build_visible_tokens_with_mapping(original_tokens, consumed_mask, placeholder)
    working_query = _rebuild_query_from_tokens(visible_tokens)
    stage_info: dict[str, Any] = {"stage_name": stage_name, "working_query": working_query, "anchor_attempts": [], "chosen_match": None, "deferred_matches": []}
    for start, token in enumerate(original_doc):
        if consumed_mask[start] or token.is_punct or token.pos_ not in allowed_seed_pos:
            continue
        anchor_attempt: dict[str, Any] = {"stage_name": stage_name, "anchor_token_index": start, "anchor_token_text": token.text, "anchor_token_pos": token.pos_, "tried_phrases": [], "selected_match": None}
        contiguous_end = start
        while contiguous_end < len(original_tokens) and not consumed_mask[contiguous_end] and not original_doc[contiguous_end].is_punct:
            contiguous_end += 1
        contiguous_count = contiguous_end - start
        max_len_here = min(max_anchor_span, contiguous_count)
        for span_len in range(max_len_here, 0, -1):
            raw_anchor = " ".join(original_tokens[start:start + span_len])
            canonical_anchor = canonicalize_title_text(raw_anchor)
            det_stripped_anchor = strip_det_tokens_from_canonical_text(_strip_det_tokens_from_span(original_tokens, original_doc, start, start + span_len - 1))
            candidates = list(det_stripped_title_map.get(det_stripped_anchor, [])) if det_stripped_anchor else []
            phrase_attempt: dict[str, Any] = {"stage_name": stage_name, "span_len": span_len, "raw_anchor": raw_anchor, "canonical_anchor": canonical_anchor, "det_stripped_anchor": det_stripped_anchor, "candidate_count": len(candidates), "candidate_titles": candidates[:20], "aligned_match": None}
            if not candidates:
                anchor_attempt["tried_phrases"].append(phrase_attempt)
                continue
            visible_start = original_to_visible.get(start)
            context_reasons = _context_reasons_for_visible_start(visible_tokens, visible_start, placeholder) if visible_start is not None else []
            for candidate_title in sorted(candidates, key=lambda t: (-len(t.split()), -len(t), t)):
                alias_matched_token_count = len(det_stripped_anchor.split())
                qid_list = canonical_title_map.get(candidate_title, [])
                qid = qid_list[0] if qid_list else ""
                qid_short = qid if str(qid).startswith("Q") else str(qid).rsplit("/", 1)[-1]
                matched_title = _resolve_raw_title_for_canonical_qid(resolver, candidate_title, qid_short) if qid_short else candidate_title
                short_match_needs_context = alias_matched_token_count < 2
                accepted = alias_matched_token_count >= 2 or bool(context_reasons)
                score_reasons = ["exact DET-stripped alias match"] if alias_matched_token_count >= 2 else (["short DET-stripped alias match rescued by title context", *context_reasons] if context_reasons else ["short DET-stripped alias match without title context → defer"])
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
                    "candidate_anchor_end": max(0, alias_matched_token_count - 1),
                    "matched_canonical_title": candidate_title,
                    "matched_title": matched_title,
                    "matched_token_count": len(candidate_title.split()),
                    "alias_matched_token_count": alias_matched_token_count,
                    "non_det_word_count": alias_matched_token_count,
                    "match_tokens_no_det": _tokens_for_later_disambiguation(raw_anchor=raw_anchor, det_stripped_anchor=det_stripped_anchor, candidate_title=candidate_title),
                    "qid": qid_short,
                    "all_qids": [q if str(q).startswith("Q") else str(q).rsplit("/", 1)[-1] for q in qid_list],
                    "wikidata_url": f"https://www.wikidata.org/wiki/{qid_short}" if qid_short else None,
                    "score": 1.0 if accepted else 0.5,
                    "score_reasons": score_reasons,
                    "visible_start_token": visible_start,
                }
                if accepted:
                    phrase_attempt["aligned_match"] = match_payload
                    anchor_attempt["selected_match"] = match_payload
                    stage_info["chosen_match"] = match_payload
                    anchor_attempt["tried_phrases"].append(phrase_attempt)
                    stage_info["anchor_attempts"].append(anchor_attempt)
                    return stage_info
                if short_match_needs_context:
                    deferred_payload = dict(match_payload)
                    deferred_payload["defer_reason"] = "short exact DET-stripped alias match without title context"
                    stage_info["deferred_matches"].append(deferred_payload)
            anchor_attempt["tried_phrases"].append(phrase_attempt)
        stage_info["anchor_attempts"].append(anchor_attempt)
    return stage_info


def _run_fuzzy_det_stripped_stage(*, stage_name: str, original_doc, original_tokens: list[str], original_pos_payload: list[dict[str, Any]], consumed_mask: list[bool], max_anchor_span: int, det_stripped_title_map: dict[str, list[str]], canonical_title_map: dict[str, list[str]], resolver, placeholder: str, allowed_seed_pos: set[str], title_token_choices: list[str]) -> dict[str, Any]:
    visible_tokens, original_to_visible = _build_visible_tokens_with_mapping(original_tokens, consumed_mask, placeholder)
    working_query = _rebuild_query_from_tokens(visible_tokens)
    stage_info: dict[str, Any] = {"stage_name": stage_name, "working_query": working_query, "anchor_attempts": [], "chosen_match": None}
    det_stripped_keys = list(det_stripped_title_map.keys())
    for start, token in enumerate(original_doc):
        if consumed_mask[start] or token.is_punct or token.pos_ not in allowed_seed_pos:
            continue
        anchor_attempt: dict[str, Any] = {"stage_name": stage_name, "anchor_token_index": start, "anchor_token_text": token.text, "anchor_token_pos": token.pos_, "tried_phrases": [], "selected_match": None}
        contiguous_end = start
        while contiguous_end < len(original_tokens) and not consumed_mask[contiguous_end] and not original_doc[contiguous_end].is_punct:
            contiguous_end += 1
        contiguous_count = contiguous_end - start
        max_len_here = min(max_anchor_span, contiguous_count)
        for span_len in range(max_len_here, 0, -1):
            raw_anchor = " ".join(original_tokens[start:start + span_len])
            canonical_anchor = canonicalize_title_text(raw_anchor)
            det_stripped_anchor = strip_det_tokens_from_canonical_text(_strip_det_tokens_from_span(original_tokens, original_doc, start, start + span_len - 1))
            typo_suspicions = _detect_typo_suspicions_in_text(det_stripped_anchor, title_token_choices) if det_stripped_anchor else []
            fuzzy_alias_hits = _extract_fuzzy_choices(det_stripped_anchor, det_stripped_keys, limit=8, score_cutoff=88.0) if typo_suspicions and det_stripped_anchor else []
            phrase_attempt: dict[str, Any] = {"stage_name": stage_name, "span_len": span_len, "raw_anchor": raw_anchor, "canonical_anchor": canonical_anchor, "det_stripped_anchor": det_stripped_anchor, "typo_suspicions": typo_suspicions, "fuzzy_alias_hits": [{"alias_key": c, "score": round(float(s), 2)} for c, s in fuzzy_alias_hits], "aligned_match": None}
            if not fuzzy_alias_hits:
                anchor_attempt["tried_phrases"].append(phrase_attempt)
                continue
            candidate_key, fuzzy_score = fuzzy_alias_hits[0]
            candidates = list(det_stripped_title_map.get(candidate_key, []))
            if not candidates:
                anchor_attempt["tried_phrases"].append(phrase_attempt)
                continue
            candidate_title = sorted(candidates, key=lambda t: (-len(t.split()), -len(t), t))[0]
            qid_list = canonical_title_map.get(candidate_title, [])
            qid = qid_list[0] if qid_list else ""
            qid_short = qid if str(qid).startswith("Q") else str(qid).rsplit("/", 1)[-1]
            matched_title = _resolve_raw_title_for_canonical_qid(resolver, candidate_title, qid_short) if qid_short else candidate_title
            score_reasons = [f"fuzzy DET-stripped alias match (score {round(float(fuzzy_score), 2)})", "typo signal triggered fuzzy fallback"]
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
                "visible_start_token": original_to_visible.get(start),
                "fuzzy_score": round(float(fuzzy_score), 2),
                "typo_suspicions": typo_suspicions,
            }
            phrase_attempt["aligned_match"] = match_payload
            anchor_attempt["selected_match"] = match_payload
            stage_info["chosen_match"] = match_payload
            anchor_attempt["tried_phrases"].append(phrase_attempt)
            stage_info["anchor_attempts"].append(anchor_attempt)
            return stage_info
        stage_info["anchor_attempts"].append(anchor_attempt)
    return stage_info


def debug_extract_titles(query: str, resolver, dataset_dir=None) -> dict[str, Any]:
    placeholder = "plazeholdaqqq"
    max_anchor_span = 5
    dataset_dir = dataset_dir or get_dataset_dir()
    canonical_title_map = _load_canonical_title_map(resolver, dataset_dir)
    canonical_phrase_map = _load_canonical_phrase_map(resolver, dataset_dir, max_span_len=max_anchor_span)
    det_stripped_title_map = _load_det_stripped_title_map(resolver, dataset_dir)
    title_token_choices = _build_title_token_choices(canonical_title_map, det_stripped_title_map)
    canonical_input = canonicalize_title_text(query)
    surface_input = _normalize_surface_text_preserve_punct(query)

    pos = POS()
    original_doc = pos.doc(surface_input)
    original_tokens = [_surface_token_text(token) for token in original_doc]
    original_pos_payload = _pos_token_payload(original_doc)
    consumed_mask = [False] * len(original_tokens)

    extracted_items: list[dict[str, Any]] = []
    passes: list[dict[str, Any]] = []
    deferred_short_matches: list[dict[str, Any]] = []

    stage_configs = [
        {"stage_name": "Stage 1 · NOUN/PROPN/DET exact", "runner": "phrase", "allowed_seed_pos": {"NOUN", "PROPN", "DET"}, "every_word": False, "require_context": False, "min_title_words": 1},
        {"stage_name": "Stage 2 · pattern recognition", "runner": "phrase", "allowed_seed_pos": None, "every_word": True, "require_context": True, "min_title_words": 1},
        {"stage_name": "Stage 3 · broad exact 3+ words", "runner": "phrase", "allowed_seed_pos": None, "every_word": True, "require_context": False, "min_title_words": 3},
        {"stage_name": "Stage 4 · DET-stripped alias fallback", "runner": "det_stripped", "allowed_seed_pos": {"NOUN", "PROPN", "DET"}, "every_word": False},
    ]

    while True:
        visible_tokens, _ = _build_visible_tokens_with_mapping(original_tokens, consumed_mask, placeholder)
        working_query = _rebuild_query_from_tokens(visible_tokens)
        pass_info: dict[str, Any] = {"pass_index": len(passes) + 1, "working_query": working_query, "tokens": original_pos_payload, "stage_order": [cfg["stage_name"] for cfg in stage_configs], "anchor_attempts": [], "stages": [], "match": None, "replacement_query": None}
        chosen = None
        for cfg in stage_configs:
            if cfg["runner"] == "phrase":
                stage_result = _run_phrase_stage(stage_name=cfg["stage_name"], original_doc=original_doc, original_tokens=original_tokens, original_pos_payload=original_pos_payload, consumed_mask=consumed_mask, max_anchor_span=max_anchor_span, canonical_phrase_map=canonical_phrase_map, canonical_title_map=canonical_title_map, resolver=resolver, placeholder=placeholder, allowed_seed_pos=cfg.get("allowed_seed_pos"), require_context=cfg.get("require_context", False), min_title_words=cfg.get("min_title_words", 1), every_word=cfg.get("every_word", False))
            elif cfg["runner"] == "det_stripped":
                stage_result = _run_det_stripped_stage(stage_name=cfg["stage_name"], original_doc=original_doc, original_tokens=original_tokens, original_pos_payload=original_pos_payload, consumed_mask=consumed_mask, max_anchor_span=max_anchor_span, det_stripped_title_map=det_stripped_title_map, canonical_title_map=canonical_title_map, resolver=resolver, placeholder=placeholder, allowed_seed_pos=cfg["allowed_seed_pos"])
            else:
                stage_result = _run_fuzzy_det_stripped_stage(stage_name=cfg["stage_name"], original_doc=original_doc, original_tokens=original_tokens, original_pos_payload=original_pos_payload, consumed_mask=consumed_mask, max_anchor_span=max_anchor_span, det_stripped_title_map=det_stripped_title_map, canonical_title_map=canonical_title_map, resolver=resolver, placeholder=placeholder, allowed_seed_pos=cfg["allowed_seed_pos"], title_token_choices=title_token_choices)
            pass_info["stages"].append(stage_result)
            pass_info["anchor_attempts"].extend(stage_result.get("anchor_attempts", []))
            deferred_short_matches.extend(stage_result.get("deferred_matches", []))
            if stage_result.get("chosen_match") is not None:
                chosen = stage_result["chosen_match"]
                break
        if chosen is None:
            passes.append(pass_info)
            break
        pass_info["match"] = chosen
        extracted_items.append({"title": chosen["matched_title"], "qid": chosen["qid"], "wikidata_url": chosen["wikidata_url"], "score": chosen["score"]})
        for idx in range(chosen["message_start_token"], chosen["message_end_token"] + 1):
            consumed_mask[idx] = True
        pass_info["replacement_query"] = _build_placeholder_view_from_consumed(original_tokens, consumed_mask, placeholder)
        passes.append(pass_info)

    return {
        "original_query": query,
        "canonical_input_query": canonical_input,
        "placeholder": placeholder,
        "max_anchor_span": max_anchor_span,
        "stage_order": [cfg["stage_name"] for cfg in stage_configs],
        "context_patterns": [label for _pattern, label in _TITLE_CONTEXT_PATTERNS] + ["recommend ... like", "and/or after title context", "comma-separated title list after trigger", "title list continuation after trigger"],
        "extracted_items": extracted_items,
        "deferred_short_matches": [match for idx, match in enumerate(deferred_short_matches) if _deferred_match_key(match) not in {_deferred_match_key(prev) for prev in deferred_short_matches[:idx]}],
        "final_query": _build_placeholder_view_from_consumed(original_tokens, consumed_mask, placeholder),
        "passes": passes,
        "title_count": len(extracted_items),
        "canonical_dictionary_title_count": len(canonical_title_map),
        "canonical_phrase_dictionary_count": len(canonical_phrase_map),
        "det_stripped_dictionary_count": len(det_stripped_title_map),
        "title_token_choice_count": len(title_token_choices),
        "fuzzy_backend": "disabled",
    }


def extract_titles(query: str, resolver, dataset_dir=None) -> list[str]:
    debug = debug_extract_titles(query, resolver, dataset_dir=dataset_dir)
    return [str(item.get("title") or "") for item in debug.get("extracted_items", []) if item.get("title")]


def replace_extracted_titles_with_placeholder(query: str, titles: list[str], placeholder: str = "  ") -> str:
    result = (query or "").lower()
    for title in sorted([t.lower() for t in titles if t], key=lambda s: (-len(s.split()), -len(s), s)):
        result = result.replace(title, placeholder)
    return result
