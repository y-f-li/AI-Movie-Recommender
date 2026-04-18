from modules.nlp.pos import POS
from modules.paths import get_dataset_dir
from modules.title_canonicalizer import canonicalize_title_text
from modules.title_extraction_latest import debug_extract_titles as debug_extract_titles_latest, replace_extracted_titles_with_placeholder
from utils.message_blocks import debug_block

import re
from pathlib import Path

ENTITY_CARRIER_WORDS = {"film", "films", "movie", "movies"}
ENTITY_CONFOUNDING_WORDS = ENTITY_CARRIER_WORDS | {"recommend", "other", "similar", "movietitle", "movieentity", "show", "shows", "drama", "dramas"}
SCI_FI_SIGNAL_ALIASES = {"sci-fi", "scifi"}
RELEASE_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")

class Extractor():
    def __init__(self, id_rel_dict, ent_id_dict, id_movie_dict, entity_synonym_matcher=None):
        self.id_rel_dict = id_rel_dict
        self.ent_id_dict = ent_id_dict
        self.id_movie_dict = id_movie_dict
        self.entity_synonym_matcher = entity_synonym_matcher
        self.common_english_words = self._load_common_english_words()
        
    def dictionary_lookup(self, q, dictionary):
        result = []
        for _, value in dictionary.items():
            pattern = r"\b" + re.escape(value) + r"\b"
            if re.search(pattern, q, flags=re.IGNORECASE):
                result.append(value)
        return result

    def rfallback(self, q):
        from modules.private_data.fallback_rel_dict import fallback_rel_dict
        relation = ""
        pos = POS()
        doc = pos.doc(q)
        for item in doc:
            if item.pos_ == "VERB":
                if item.lemma_ in fallback_rel_dict:
                    relation = fallback_rel_dict[item.lemma_]
        return (relation if len(relation) > 0 else None)

    def rhungry(self, q):

        ignore = {"movie", "film", "MovieTitle"}
        signals = []
        candidates = []
        pos = POS()
        doc = pos.doc(q)
        for item in doc:
            if item.pos_ == "NOUN" and (item.text not in ignore):
                signals.append(item.text)
        for sig in signals:
            for _, rel in self.id_rel_dict.items():
                if sig in rel.split():
                    candidates.append(rel)
        for can in candidates:
            if can in q:
                return can
        return None

    def rhelper(self, q):
        """
        look for relation from the relation dict
        If we can't find the relationship
        check if date applies first, then
        go through the pos_ and look up the lemma for verbs
        """
        relation = None
        from modules.pattern_library import PatternLibrary
        from modules.private_data.rel_pattern_dict import rel_pattern_dict
        pl = PatternLibrary()
        for pattern in pl.patterns:
            if re.search(pattern[0], q):
                relation = pattern[1]
                return rel_pattern_dict[relation]

        relation = self.dictionary_lookup(q, self.id_rel_dict)
        if len(relation) > 0:
            result = []
            for item in relation:
                if item not in self.ent_id_dict:
                    result.append(item)
            relation = result if result else relation
            relation.sort(key=lambda s: len(s.split()), reverse=True)
            return relation[0]
        elif self.rfallback(q):
            relation = self.rfallback(q)
            return relation
        elif self.rhungry(q):
            return self.rhungry(q)
        if len(relation)==0:
            return None

    def extract_one_hop(self, q):
        titles = [str(item.get("title") or "").lower() for item in debug_extract_titles_latest(q or "", self._resolver_proxy()).get("extracted_items", []) if item.get("title")]
        q2 = replace_extracted_titles_with_placeholder(q or "", titles, "  ")
        return titles, self.rhelper(q2)

    # recommender + multimedia

    def regex_title(self, q, dictionary, placeholder = "  "):
        q = q.lower()
        result = []
        matches = self.dictionary_lookup(q, dictionary)
        ordered = sorted(matches, key=lambda s: len(s.split()), reverse=True)
        for mtch in ordered:
            mtch = mtch.lower()
            if mtch in q:
                q = q.replace(mtch, placeholder)
                result.append(mtch)
        return q, result


    def debug_extract_movie_titles(self, q, placeholder="[title]"):
        debug = debug_extract_titles_latest(q or "", self._resolver_proxy())
        extracted_titles = [str(item.get("title") or "") for item in debug.get("extracted_items", []) if item.get("title")]
        residual = replace_extracted_titles_with_placeholder(q or "", extracted_titles, f" {placeholder} ")
        return {
            "original_query": q or "",
            "lowercased_query": (q or "").lower(),
            "canonical_query": debug.get("canonical_input_query"),
            "extracted_titles": extracted_titles,
            "residual_query": re.sub(r"\s+", " ", residual).strip(),
            "title_count": len(extracted_titles),
            "latest_debug": debug,
        }

    def _resolver_proxy(self):
        class _ResolverProxy:
            pass
        proxy = _ResolverProxy()
        proxy.t_q_dict = {str(v).lower(): str(k).rsplit("/", 1)[-1] for k, v in self.id_movie_dict.items() if v is not None}
        return proxy

    def qfiltered(self, q):
        result = ""
        sig_pos = {"NOUN", "PROPN", "ADJ"}
        pos = POS()
        doc = pos.doc(q)
        for item in doc:
            if item.pos_ not in sig_pos:
                result += "  "
            else:
                result += item.text + " "
        return result

    def _normalize_signal_sequence_aliases(self, tokens):
        normalized_tokens = []
        cleaned_tokens = [str(tok).strip() for tok in (tokens or []) if str(tok).strip()]
        index = 0
        while index < len(cleaned_tokens):
            current = cleaned_tokens[index]
            current_lower = current.lower()
            next_lower = cleaned_tokens[index + 1].lower() if index + 1 < len(cleaned_tokens) else None

            if current_lower in SCI_FI_SIGNAL_ALIASES:
                normalized_tokens.append("science fiction")
                index += 1
                continue

            if current_lower == "sci" and next_lower == "fi":
                normalized_tokens.append("science fiction")
                index += 2
                continue

            normalized_tokens.append(current)
            index += 1

        return normalized_tokens

    def _collect_signal_token_sequences(self, q):
        sig_pos = {"NOUN", "PROPN", "ADJ"}
        sequences = []
        pos = POS()
        doc = pos.doc(q or "")
        for sent in doc.sents:
            sentence_tokens = [token for token in sent if not token.is_space]
            sent_tokens = []
            index = 0
            while index < len(sentence_tokens):
                token = sentence_tokens[index]
                token_text = str(token.text or "").strip()
                if not token_text:
                    index += 1
                    continue
                lower = token_text.lower()
                next_lower = str(sentence_tokens[index + 1].text or "").strip().lower() if index + 1 < len(sentence_tokens) else None

                if lower in SCI_FI_SIGNAL_ALIASES:
                    sent_tokens.append("science fiction")
                    index += 1
                    continue

                if lower == "sci" and next_lower == "fi":
                    sent_tokens.append("science fiction")
                    index += 2
                    continue

                if token.pos_ in sig_pos or lower in ENTITY_CARRIER_WORDS:
                    sent_tokens.append(token_text)
                index += 1

            sent_tokens = self._normalize_signal_sequence_aliases(sent_tokens)
            if sent_tokens:
                sequences.append(sent_tokens)
        return sequences

    def _raw_candidate_subspans_from_tokens(self, tokens, max_span=5):
        cleaned_tokens = [str(tok).strip() for tok in (tokens or []) if str(tok).strip()]
        if not cleaned_tokens:
            return []

        max_span = max(1, min(int(max_span), len(cleaned_tokens)))
        spans = []
        seen = set()

        for span_len in range(max_span, 0, -1):
            for start in range(0, len(cleaned_tokens) - span_len + 1):
                span_tokens = cleaned_tokens[start:start + span_len]
                span_text = " ".join(span_tokens)
                normalized_span = re.sub(r"\s+", " ", span_text).strip().lower()
                if not normalized_span:
                    continue
                if normalized_span in seen:
                    continue
                seen.add(normalized_span)
                spans.append(normalized_span)
        return spans

    def _strip_confounding_words_from_text(self, text):
        tokens = [tok for tok in re.findall(r"[A-Za-z0-9']+", text or "") if tok.lower() not in ENTITY_CONFOUNDING_WORDS]
        return " ".join(tokens).strip().lower()

    def _filter_clear_signal_spans(self, spans):
        cleaned_spans = []
        seen = set()
        for span in spans or []:
            cleaned_span = self._strip_confounding_words_from_text(span)
            cleaned_span = re.sub(r"\s+", " ", cleaned_span).strip().lower()
            if not cleaned_span or cleaned_span in seen:
                continue
            if not self.clear_signal(cleaned_span):
                continue
            seen.add(cleaned_span)
            cleaned_spans.append(cleaned_span)
        return cleaned_spans

    def _span_token_list(self, text):
        return [tok for tok in re.findall(r"[A-Za-z0-9']+", text or "") if tok]

    def _span_contains_subspan(self, container_span, subspan):
        container_tokens = [tok.lower() for tok in self._span_token_list(container_span)]
        subspan_tokens = [tok.lower() for tok in self._span_token_list(subspan)]
        if not container_tokens or not subspan_tokens:
            return False
        if len(subspan_tokens) >= len(container_tokens):
            return False
        window = len(subspan_tokens)
        for start in range(0, len(container_tokens) - window + 1):
            if container_tokens[start:start + window] == subspan_tokens:
                return True
        return False

    def _span_has_exact_entity_match(self, span):
        for form in self._entity_candidate_forms(span):
            if form in self.ent_id_dict:
                return True
        return False

    def _suppress_subspans_under_longest_exact_matches(self, spans):
        candidate_spans = [span for span in (spans or []) if span]
        exact_match_spans = [span for span in candidate_spans if self._span_has_exact_entity_match(span)]
        if not exact_match_spans:
            return candidate_spans

        kept_spans = []
        for span in candidate_spans:
            covered_by_longer_exact_match = False
            for exact_span in exact_match_spans:
                if exact_span == span:
                    continue
                if not self._span_contains_subspan(exact_span, span):
                    continue
                covered_by_longer_exact_match = True
                break
            if not covered_by_longer_exact_match:
                kept_spans.append(span)
        return kept_spans

    def _candidate_subspans_from_tokens(self, tokens, max_span=5):
        raw_spans = self._raw_candidate_subspans_from_tokens(tokens, max_span=max_span)
        filtered_spans = self._filter_clear_signal_spans(raw_spans)
        return self._suppress_subspans_under_longest_exact_matches(filtered_spans)

    def clear_signal(self, q):
        for item in q.split():
            if item.lower() not in ENTITY_CONFOUNDING_WORDS:
                return True
        return False

    def noun_lemmaize(self, q):
        pos = POS()
        doc = pos.doc(q)
        for item in doc:
            if item.pos_ in {"NOUN", "PROPN"}:
                q = q.replace(item.text, item.lemma_)
        return q

    def strip_entity_carrier_words(self, text):
        tokens = [t for t in re.findall(r"[A-Za-z0-9']+", text or "") if t.lower() not in ENTITY_CARRIER_WORDS]
        return " ".join(tokens).strip().lower()

    def _load_common_english_words(self):
        dataset_dir = get_dataset_dir()
        candidates = [dataset_dir / "top-1000-nouns.txt", Path(__file__).resolve().parents[2] / "dataset" / "top-1000-nouns.txt"]
        words = set()
        for path in candidates:
            try:
                if not path.exists():
                    continue
                for raw_line in path.read_text(encoding="utf-8").splitlines():
                    cleaned = canonicalize_title_text(raw_line)
                    for token in cleaned.split():
                        if token:
                            words.add(token)
                if words:
                    break
            except Exception:
                continue
        return words

    def _canonical_word_tokens(self, text):
        return [tok for tok in canonicalize_title_text(text or "").split() if tok]

    def _entity_token_set(self, entities):
        token_set = set()
        for entity in entities or []:
            token_set.update(self._canonical_word_tokens(entity))
        return token_set

    def _dedupe_preserve_order(self, values):
        return list(dict.fromkeys(v for v in values if v))

    def _extract_release_year_entities(self, text):
        years = []
        for match in RELEASE_YEAR_PATTERN.finditer(text or ""):
            year = str(match.group(0) or "").strip()
            if year:
                years.append(year)
        return self._dedupe_preserve_order(years)

    def _resolve_deferred_short_title_candidates(self, deferred_matches, entities):
        entity_tokens = self._entity_token_set(entities)
        rescued_titles = []
        clarify_titles = []
        seen_clarify = set()

        for match in deferred_matches or []:
            candidate_title = str(match.get("matched_title") or match.get("anchor_raw_span") or "").strip()
            match_tokens = [tok for tok in match.get("match_tokens_no_det", []) if tok]
            if not candidate_title or not match_tokens:
                continue

            unmatched_tokens = [tok for tok in match_tokens if tok not in entity_tokens]
            uncommon_unmatched = [tok for tok in unmatched_tokens if tok not in self.common_english_words]

            if uncommon_unmatched:
                rescued_titles.append(candidate_title.lower())
                continue

            if candidate_title not in seen_clarify:
                seen_clarify.add(candidate_title)
                clarify_titles.append(candidate_title)

        return self._dedupe_preserve_order(rescued_titles), clarify_titles

    def _entity_candidate_forms(self, text):
        forms = []
        if text and text.strip():
            forms.append(text.strip().lower())
        stripped = self.strip_entity_carrier_words(text)
        if stripped:
            forms.append(stripped)
        lemma = self.noun_lemmaize(text or "").strip().lower()
        if lemma:
            forms.append(lemma)
        lemma_stripped = self.strip_entity_carrier_words(lemma)
        if lemma_stripped:
            forms.append(lemma_stripped)
        # simple film/movie variant bridge before stripping
        swapped = (text or "").replace("movie", "film").replace("movies", "films")
        if swapped and swapped.strip():
            forms.append(swapped.strip().lower())
        swapped2 = (text or "").replace("film", "movie").replace("films", "movies")
        if swapped2 and swapped2.strip():
            forms.append(swapped2.strip().lower())
        out = []
        seen = set()
        for f in forms:
            f = re.sub(r"\s+", " ", f).strip()
            if f and f not in seen:
                seen.add(f)
                out.append(f)
        return out

    def _debug_synonym_trace(self, text):
        matcher = self.entity_synonym_matcher
        normalized_text = re.sub(r"\s+", " ", str(text or "").strip().lower())
        if matcher is None:
            return {
                "normalized_text": normalized_text,
                "exact_alias_hit": None,
                "resolved_label": None,
                "resolved_score": 0.0,
                "matched_labels": [],
            }

        exact_alias_hit = matcher.alias_to_canonical.get(normalized_text)
        resolved_label, resolved_score = matcher.resolve_label(text)
        matched_labels = matcher.match_labels_from_text(text)

        return {
            "normalized_text": normalized_text,
            "exact_alias_hit": exact_alias_hit,
            "resolved_label": resolved_label,
            "resolved_score": float(resolved_score or 0.0),
            "matched_labels": matched_labels,
        }

    def _log_entity_resolution_trace(self, token_sequences, debug_sequences):
        with debug_block("recommendation-entity-resolution"):
            print(f"signal sequences: {debug_sequences}")
            for seq_index, token_sequence in enumerate(token_sequences, start=1):
                sequence_text = " ".join(token_sequence)
                raw_candidate_spans = self._raw_candidate_subspans_from_tokens(token_sequence)
                filtered_candidate_spans = self._filter_clear_signal_spans(raw_candidate_spans)
                surviving_candidate_spans = self._suppress_subspans_under_longest_exact_matches(filtered_candidate_spans)

                print(f"[EntityTrace][Sequence {seq_index}] raw signal sequence: {sequence_text}")
                print(f"[EntityTrace][Sequence {seq_index}] raw subspans: {raw_candidate_spans}")
                print(
                    f"[EntityTrace][Sequence {seq_index}] subspans after confounding-word stripping/filter: "
                    f"{filtered_candidate_spans}"
                )
                print(
                    f"[EntityTrace][Sequence {seq_index}] subspans after exact-match longest-span suppression: "
                    f"{surviving_candidate_spans}"
                )

                if not surviving_candidate_spans:
                    cleaned = self._strip_confounding_words_from_text(sequence_text)
                    print(f"[EntityTrace][Sequence {seq_index}] no surviving subspans; stripped leftover candidate: {cleaned!r}")
                    continue

                sequence_matches = []

                for span in surviving_candidate_spans:
                    candidate_forms = self._entity_candidate_forms(span)
                    direct_matches = [form for form in candidate_forms if form in self.ent_id_dict]
                    synonym_debug = [self._debug_synonym_trace(form) for form in candidate_forms]

                    synonym_layer_labels = []
                    for trace in synonym_debug:
                        for label in trace.get("matched_labels", []):
                            if label not in synonym_layer_labels:
                                synonym_layer_labels.append(label)

                    span_matches = []
                    for label in direct_matches + synonym_layer_labels:
                        if label in self.ent_id_dict and label not in span_matches:
                            span_matches.append(label)
                    for label in span_matches:
                        if label not in sequence_matches:
                            sequence_matches.append(label)

                    print(f"[EntityTrace][Sequence {seq_index}][Span] {span!r}")
                    print(f"  candidate forms: {candidate_forms}")
                    print(f"  direct dictionary matches: {direct_matches}")
                    for trace in synonym_debug:
                        print(
                            f"  synonym layer for {trace['normalized_text']!r}: "
                            f"exact_alias_hit={trace['exact_alias_hit']} | "
                            f"resolved_label={trace['resolved_label']} | resolved_score={trace['resolved_score']:.3f} | "
                            f"matched_labels={trace['matched_labels']}"
                        )
                    print(f"  final matches kept for span: {span_matches}")

                print(f"[EntityTrace][Sequence {seq_index}] final matches kept for sequence: {sequence_matches}")

    def extract_rec_mulm(self, q):
        resolver = self._resolver_proxy()
        title_debug = debug_extract_titles_latest(q or "", resolver)
        movie_titles = [str(item.get("title") or "").lower() for item in title_debug.get("extracted_items", []) if item.get("title")]
        q1 = replace_extracted_titles_with_placeholder(q or "", movie_titles, "  ")
        release_year_entities = self._extract_release_year_entities(q1)

        token_sequences = self._collect_signal_token_sequences(q1)
        debug_sequences = [" ".join(seq) for seq in token_sequences]
        self._log_entity_resolution_trace(token_sequences, debug_sequences)

        ent = list(release_year_entities)
        left_over = []
        for token_sequence in token_sequences:
            sequence_text = " ".join(token_sequence)
            candidate_spans = self._candidate_subspans_from_tokens(token_sequence)
            matched_here = []

            for span in candidate_spans:
                candidate_forms = self._entity_candidate_forms(span)

                for form in candidate_forms:
                    if form in self.ent_id_dict and form not in matched_here:
                        matched_here.append(form)

                if self.entity_synonym_matcher is not None:
                    for form in candidate_forms:
                        for label in self.entity_synonym_matcher.match_labels_from_text(form):
                            if label in self.ent_id_dict and label not in matched_here:
                                matched_here.append(label)

            if matched_here:
                ent.extend(matched_here)
            else:
                cleaned = self._strip_confounding_words_from_text(sequence_text)
                if cleaned and self.clear_signal(cleaned):
                    left_over.append(cleaned)

        ent = self._dedupe_preserve_order(ent)
        left_over = self._dedupe_preserve_order(left_over)
        rescued_titles, clarify_titles = self._resolve_deferred_short_title_candidates(title_debug.get("deferred_short_matches", []), ent)
        movie_titles = self._dedupe_preserve_order(movie_titles + rescued_titles)

        if left_over and not (ent or movie_titles or clarify_titles):
            return None

        return {
            "titles": movie_titles,
            "entities": ent,
            "left_over": left_over,
            "clarify_titles": clarify_titles,
            "rescued_short_titles": rescued_titles,
            "deferred_short_matches": title_debug.get("deferred_short_matches", []),
            "title_debug": title_debug,
            "signal_sequences": debug_sequences,
        }
