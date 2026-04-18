import json
import random
import re
from modules.paths import get_dataset_dir
from modules.popularity_tier import POPULARITY_TIER_RELATION, popularity_weight_from_value, augment_movie_attrs_with_popularity, augment_attr_index_with_popularity
from modules.recommender.recommender_file_helper import RatingsFileProcessor
from modules.title_resolver import TitleResolver
from utils.message_blocks import debug_block
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd

class KNNRecommender:
    def __init__(
        self,
        rating_matrix: pd.DataFrame,
        content_recommendations: list[str] | None = None,
        n_neighbors: int = 20,
    ):
        """
        rating_matrix:
            rows   = movie_id (index)
            cols   = user_id
            values = centered ratings (or plain ratings)
        """
        # Always keep full matrix so liked movies are present
        self.rmx = rating_matrix

        # Optional candidate set for neighbors
        self.candidate_set = set(content_recommendations) if content_recommendations is not None else None

        self.movie_ids = self.rmx.index.to_numpy()
        self.ml_csr = csr_matrix(self.rmx.values)

        self.knn = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_neighbors=n_neighbors,
        )
        self.knn.fit(self.ml_csr)

    def recommend(self, liked_list, per_movie_neighbors: int = 5, top_k: int = 20):
        liked_set = set(liked_list)
        candidate_scores: dict[str, float] = {}

        for movie_id in liked_list:
            if movie_id not in self.rmx.index:
                continue

            row_idx = self.rmx.index.get_loc(movie_id)
            distances, indices = self.knn.kneighbors(
                self.ml_csr[row_idx],
                n_neighbors=per_movie_neighbors + 1,
            )

            distances = distances[0]
            indices = indices[0]

            for dist, idx in zip(distances, indices):
                neighbor_id = self.movie_ids[idx]

                if neighbor_id == movie_id or neighbor_id in liked_set:
                    continue

                # NEW: restrict to candidate set if provided
                if self.candidate_set is not None and neighbor_id not in self.candidate_set:
                    continue

                similarity = 1.0 - dist
                candidate_scores[neighbor_id] = candidate_scores.get(neighbor_id, 0.0) + similarity

        ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        recommended_ids = [m_id for m_id, _ in ranked[:top_k]]

        with debug_block():
            print(f"The recommended movie id's are:\n{recommended_ids}")

        return recommended_ids



class KNNRecommenderDebugger:
    """
    This class is a debug helper for KNNRecommender.
    """
    def __init__(self, kr: KNNRecommender):
        self.kr = kr
        self.example_liked_list = [
            "http://www.wikidata.org/entity/Q1001102",
            "http://www.wikidata.org/entity/Q100146309 ",
            "http://www.wikidata.org/entity/Q100148104"
            ]

    def print_debug_message(self):
        """
        Prints debug messages for the KNNRecommender class.
        """
        with debug_block():
            print("recommendation:\n", self.kr.recommend(self.example_liked_list))

class ContentRecommender:
    GENRE_RELATION = "http://www.wikidata.org/prop/direct/P136"
    ATAI_TAG_RELATION = "http://ddis.ch/atai/tag"
    RELEASE_DATE_RELATION = "http://www.wikidata.org/prop/direct/P577"
    RELEASE_YEAR_ENTITY_PREFIX = "__release_year__:"
    POPULARITY_TIER_RELATION = POPULARITY_TIER_RELATION
    CAST_MEMBER_RELATION = "http://www.wikidata.org/prop/direct/P161"

    WEIGHT_4_RELATIONS = {
        "http://www.wikidata.org/prop/direct/P495",  # country of origin
        "http://www.wikidata.org/prop/direct/P57",   # director
        "http://www.wikidata.org/prop/direct/P136",  # genre
        "http://www.wikidata.org/prop/direct/P364",  # original language of film or tv show
        "http://ddis.ch/atai/tag",                   # tag
    }
    WEIGHT_3_RELATIONS = {
        "http://www.wikidata.org/prop/direct/P344",  # director of photography
        "http://www.wikidata.org/prop/direct/P86",   # composer
        "http://ddis.ch/atai/popularity_tier",       # popularity tier
    }
    TOP_CELEB_FILENAME = "top-500-celeb.json"

    def __init__(self, movie_attrs, rating_matrix: pd.DataFrame, attr_index=None, genre_tag_bag_index=None, id_entity_dict=None):
        self.m_attrs = movie_attrs
        self.r_m = rating_matrix
        self.attr_index = attr_index or {}
        augment_movie_attrs_with_popularity(self.m_attrs)
        augment_attr_index_with_popularity(self.attr_index, self.m_attrs)
        self.genre_tag_bag_index = genre_tag_bag_index or {}
        self.id_entity_dict = id_entity_dict or {}
        self.top_celeb_rank_lookup = self._load_top_celeb_rank_lookup()

    def _normalize_qid(self, qid: str) -> str:
        # If it's a full IRI like "http://www.wikidata.org/entity/Q123",
        # keep only the "Q123" part. Otherwise return as-is.
        if qid.startswith("http://") or qid.startswith("https://"):
            return qid.rstrip("/").split("/")[-1]
        return qid


    def _normalize_text(self, text: str) -> str:
        s = (text or "").strip().lower()
        s = s.replace("’", "'").replace("‘", "'").replace('“', '"').replace('”', '"')
        s = s.replace("_", " ")
        s = s.replace("-", " ")
        s = " ".join(s.split())
        return s

    def _bag_tokens(self, text: str) -> list[str]:
        stop = {"the", "of", "a", "an", "and", "or", "film", "films", "movie", "movies"}
        return [t for t in re.findall(r"[a-z0-9]+", self._normalize_text(text)) if t and t not in stop]

    def _carrier_stripped_label(self, text: str) -> str:
        return self._normalize_text(" ".join(self._bag_tokens(text)))

    def _load_top_celeb_rank_lookup(self) -> dict[str, int]:
        dataset_path = get_dataset_dir() / self.TOP_CELEB_FILENAME
        if not dataset_path.exists():
            return {}
        try:
            data = json.loads(dataset_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        lookup: dict[str, int] = {}
        for raw_name, raw_rank in data.items():
            key = self._normalize_text(str(raw_name or ""))
            if not key:
                continue
            try:
                lookup[key] = int(raw_rank)
            except Exception:
                continue
        return lookup

    def _entity_label_for_raw_value(self, raw_value: str) -> str:
        raw = str(raw_value or "").strip()
        if not raw:
            return ""
        norm_qid = self._normalize_qid(raw)
        label = self.id_entity_dict.get(raw) or self.id_entity_dict.get(norm_qid) or raw
        return str(label or "")

    def _cast_member_weight_for_value(self, raw_value: str) -> float:
        label = self._entity_label_for_raw_value(raw_value)
        normalized_label = self._normalize_text(label)
        if not normalized_label:
            return 1.0
        rank = self.top_celeb_rank_lookup.get(normalized_label)
        if rank is None:
            return 1.0
        if 0 <= int(rank) <= 300:
            return 3.0
        if 301 <= int(rank) <= 500:
            return 2.0
        return 1.0

    def _profile_weight_for_attr_value(self, attr: str, value: str, *, weighted_title_mode: bool) -> float:
        if attr == self.CAST_MEMBER_RELATION:
            return self._cast_member_weight_for_value(value)
        if weighted_title_mode:
            if attr in self.WEIGHT_4_RELATIONS:
                return 4.0
            if attr in self.WEIGHT_3_RELATIONS:
                return 3.0
        return 1.0

    def _is_release_year_entity(self, qid: str) -> bool:
        return isinstance(qid, str) and qid.startswith(self.RELEASE_YEAR_ENTITY_PREFIX)

    def _extract_release_year_literal(self, qid: str) -> str:
        if not self._is_release_year_entity(qid):
            return ""
        return str(qid.split(":", 1)[1] or "").strip()

    def _release_year_from_raw_value(self, raw_value: str) -> str:
        raw = str(raw_value or "").strip()
        match = re.search(r"(?:19|20)\d{2}", raw)
        return match.group(0) if match else ""

    def _lookup_bag_phrase_values(self, attr: str, phrase: str) -> set[str]:
        attrs = self.genre_tag_bag_index.get("attrs", {}) if self.genre_tag_bag_index else {}
        attr_info = attrs.get(attr, {})
        phrase_lookup = attr_info.get("phrase_lookup", {})
        phrase = self._carrier_stripped_label(phrase)
        if not phrase:
            return set()
        return {str(rv) for rv in phrase_lookup.get(phrase, [])}

    def _lookup_bag_token_values(self, attr: str, tokens: set[str]) -> set[str]:
        attrs = self.genre_tag_bag_index.get("attrs", {}) if self.genre_tag_bag_index else {}
        attr_info = attrs.get(attr, {})
        token_lookup = attr_info.get("token_lookup", {})
        out: set[str] = set()
        for tok in tokens:
            if not tok:
                continue
            for rv in token_lookup.get(tok, []):
                out.add(str(rv))
        return out

    def _is_bag_target_attr(self, attr: str) -> bool:
        return attr in {self.GENRE_RELATION, self.ATAI_TAG_RELATION}

    def build_preference_profile(
        self,
        liked_movie_list: list[str] | None,
        entity_qids: list[str] | None = None,
    ) -> dict[tuple[str, str], float]:
        """
        Build a preference profile from:
        - liked_movie_list: movies the user likes
        - entity_qids: explicitly mentioned entities (actors, genres, etc.)

        Each appearance (either via movie attributes or via entity list)
        adds +1 to that (attr, value) pair.
        """
        preference_profile: dict[tuple[str, str], float] = {}
        liked_movie_list = liked_movie_list or []
        entity_qids = entity_qids or []
        weighted_title_mode = bool(liked_movie_list)

        # 1) from liked movies
        # When recommendation messages contain movie titles, do not treat repeated
        # attribute/value pairs across liked movies equally. Instead:
        # - if an (attr, value) appears in multiple liked movies, boost it with
        #   base_weight * (5 ** (movie_count_with_attr - 1))
        # - if it appears in only one liked movie, penalize it with
        #   base_weight * (0.8 ** total_liked_movies)
        # The multiplier is applied to the base weight, not used as a replacement.
        if liked_movie_list:
            liked_attr_counts: dict[tuple[str, str], int] = {}
            for qid in liked_movie_list:
                movie_attrs = self.m_attrs.get(qid)
                if not movie_attrs:
                    continue
                for attr, attr_set in movie_attrs.items():
                    for value in attr_set:
                        key = (attr, value)
                        liked_attr_counts[key] = liked_attr_counts.get(key, 0) + 1

            total_liked_movies = len(liked_movie_list)
            singleton_penalty = 0.8 ** total_liked_movies if total_liked_movies > 0 else 1.0

            for (attr, value), movie_count_with_attr in liked_attr_counts.items():
                base_weight = self._profile_weight_for_attr_value(attr, value, weighted_title_mode=weighted_title_mode)
                if movie_count_with_attr > 1:
                    overlap_multiplier = 5 ** (movie_count_with_attr - 1)
                else:
                    overlap_multiplier = singleton_penalty
                increment = base_weight * overlap_multiplier
                preference_profile[(attr, value)] = preference_profile.get((attr, value), 0.0) + increment

        # 2) from explicitly mentioned entities
        if entity_qids:
            for qid in entity_qids:
                for attr, raw_values in self._resolved_raw_values_for_entity_qid(qid).items():
                    for boosted_value in raw_values:
                        key = (attr, boosted_value)
                        increment = self._profile_weight_for_attr_value(attr, boosted_value, weighted_title_mode=weighted_title_mode)
                        preference_profile[key] = preference_profile.get(key, 0.0) + increment
        return preference_profile


    def _resolved_raw_values_for_entity_qid(self, qid: str) -> dict[str, set[str]]:
        """
        Resolve an entity QID/IRI to matching raw attribute values in attr_index.
        For genre/tag attributes, use the precomputed term-aware lookup artifact:
          - exact carrier-stripped phrase lookup first
          - for single-word phrases, also union token matches
          - for multi-word phrases, use token lookup only as a fallback if phrase lookup is empty
        This preserves recall for queries like "thriller" -> "action thriller" while
        keeping multi-word phrases like "science fiction" relatively precise.
        Returns {attr_iri: {raw_value, ...}}.
        """
        resolved: dict[str, set[str]] = {}
        if not self.attr_index or not qid:
            return resolved

        if self._is_release_year_entity(qid):
            target_year = self._extract_release_year_literal(qid)
            if not target_year:
                return resolved
            value_to_movies = self.attr_index.get(self.RELEASE_DATE_RELATION, {}) or {}
            matched_raw_values = {str(raw_value) for raw_value in value_to_movies.keys() if self._release_year_from_raw_value(str(raw_value)) == target_year}
            if matched_raw_values:
                resolved[self.RELEASE_DATE_RELATION] = matched_raw_values
            return resolved

        norm_qid = self._normalize_qid(qid)
        entity_label = str(self.id_entity_dict.get(qid) or self.id_entity_dict.get(norm_qid) or "")
        label_norm = self._normalize_text(entity_label)
        label_stripped = self._carrier_stripped_label(entity_label)
        label_tokens = set(self._bag_tokens(entity_label))

        for attr, value_to_movies in self.attr_index.items():
            matched_raw_values: set[str] = set()

            # Direct exact qid/raw-value matches always remain valid.
            if norm_qid in value_to_movies:
                matched_raw_values.add(norm_qid)
            for raw_value in value_to_movies.keys():
                raw_str = str(raw_value)
                if raw_str in matched_raw_values:
                    continue
                if self._normalize_qid(raw_str) == norm_qid:
                    matched_raw_values.add(raw_str)

            if self._is_bag_target_attr(attr):
                # Term-aware lookup from the new artifact.
                phrase_matches = set()
                phrase_candidates = {label_stripped, self._carrier_stripped_label(label_norm)}
                for phrase in phrase_candidates:
                    if phrase:
                        phrase_matches |= self._lookup_bag_phrase_values(attr, phrase)

                token_matches = self._lookup_bag_token_values(attr, label_tokens) if label_tokens else set()
                token_count = len(label_tokens)

                if token_count <= 1:
                    # Single-word genre/tag requests like "thriller" need both exact-phrase
                    # and token matches so compound values such as "action thriller" are included.
                    matched_raw_values |= phrase_matches
                    matched_raw_values |= token_matches
                else:
                    # Multi-word requests stay phrase-first for precision, but still fall back
                    # to token lookup when the phrase is not present in the lookup artifact.
                    if phrase_matches:
                        matched_raw_values |= phrase_matches
                    else:
                        matched_raw_values |= token_matches

            if matched_raw_values:
                resolved[attr] = {str(v) for v in matched_raw_values}

        return resolved

    def candidate_movies_for_entity_qid(self, qid: str) -> set[str]:
        """Return the set of movies satisfying a single extracted entity checkbox."""
        candidates: set[str] = set()
        for attr, raw_values in self._resolved_raw_values_for_entity_qid(qid).items():
            value_to_movies = self.attr_index.get(attr, {})
            for raw_value in raw_values:
                candidates |= set(value_to_movies.get(raw_value, set()))
        return candidates

    def _rank_candidate_pool(
        self,
        candidates: set[str],
        preference_profile: dict[tuple[str, str], float],
        liked_movie_list: list[str] | None = None,
        limit: int = 5,
    ) -> list[tuple[str, float]]:
        liked_set = set(liked_movie_list or [])
        scored: list[tuple[str, float]] = []
        for movie_id in candidates:
            if movie_id in liked_set:
                continue
            explanation = self.explain_movie_against_profile(movie_id, preference_profile)
            score = float(explanation["total_score"])
            if score > 0:
                scored.append((movie_id, score))
        scored.sort(key=lambda item: (-item[1], str(item[0])))
        return scored[:limit]

    def _popularity_tier_for_movie(self, movie_id: str) -> int:
        attrs = self.m_attrs.get(movie_id, {}) or {}
        values = list(attrs.get(POPULARITY_TIER_RELATION, set()) or [])
        if not values:
            return 1
        try:
            return max(1, int(popularity_weight_from_value(values[0])))
        except Exception:
            return 1

    def _movie_ids_grouped_by_popularity_tier(self, movie_ids, exclude: set[str] | None = None) -> dict[int, list[str]]:
        exclude = exclude or set()
        grouped: dict[int, list[str]] = {}
        for movie_id in movie_ids:
            if movie_id in exclude:
                continue
            tier = self._popularity_tier_for_movie(movie_id)
            grouped.setdefault(tier, []).append(movie_id)
        return grouped

    def _sample_random_from_highest_tiers(
        self,
        movie_ids,
        count: int,
        exclude: set[str] | None = None,
    ) -> list[str]:
        grouped = self._movie_ids_grouped_by_popularity_tier(movie_ids, exclude=exclude)
        selected: list[str] = []
        for tier in sorted(grouped.keys(), reverse=True):
            tier_movies = list(grouped[tier])
            if not tier_movies:
                continue
            remaining = count - len(selected)
            if remaining <= 0:
                break
            if len(tier_movies) <= remaining:
                random.shuffle(tier_movies)
                selected.extend(tier_movies)
            else:
                selected.extend(random.sample(tier_movies, remaining))
                break
        return selected

    def _sample_random_from_top_n_tiers(
        self,
        movie_ids,
        count: int,
        top_n_tiers: int = 2,
        exclude: set[str] | None = None,
    ) -> list[str]:
        grouped = self._movie_ids_grouped_by_popularity_tier(movie_ids, exclude=exclude)
        top_tiers = sorted(grouped.keys(), reverse=True)[:max(1, top_n_tiers)]
        pool: list[str] = []
        for tier in top_tiers:
            pool.extend(grouped.get(tier, []))
        if not pool or count <= 0:
            return []
        if len(pool) <= count:
            random.shuffle(pool)
            return pool
        return random.sample(pool, count)

    def _first_distinct_ranked_movie(self, ranked: list[tuple[str, float]], exclude: set[str] | None = None) -> str | None:
        exclude = exclude or set()
        for movie_id, _score in ranked:
            if movie_id not in exclude:
                return movie_id
        return None

    def _select_entity_only_full_intersection_movies(
        self,
        full_candidates: set[str],
        preference_profile: dict[tuple[str, str], float],
        liked_movie_list: list[str] | None,
        limit: int,
    ) -> tuple[list[str], list[tuple[str, float]]]:
        ranked_all = self._rank_candidate_pool(full_candidates, preference_profile, liked_movie_list, max(limit, len(full_candidates)))
        if len(full_candidates) <= limit:
            if ranked_all:
                return [mid for mid, _ in ranked_all[:limit]], ranked_all
            return self._sample_random_from_highest_tiers(full_candidates, limit), ranked_all

        selected: list[str] = []
        selected.extend(self._sample_random_from_highest_tiers(full_candidates, 3))
        selected.extend(self._sample_random_from_top_n_tiers(full_candidates, 1, top_n_tiers=2, exclude=set(selected)))

        ranked_pick = self._first_distinct_ranked_movie(ranked_all, exclude=set(selected))
        if ranked_pick is not None:
            selected.append(ranked_pick)

        if len(selected) < limit:
            selected.extend(self._sample_random_from_highest_tiers(full_candidates, limit - len(selected), exclude=set(selected)))

        if len(selected) < limit:
            for movie_id, _score in ranked_all:
                if movie_id in selected:
                    continue
                selected.append(movie_id)
                if len(selected) >= limit:
                    break

        deduped: list[str] = []
        seen: set[str] = set()
        for movie_id in selected:
            if movie_id in seen:
                continue
            seen.add(movie_id)
            deduped.append(movie_id)
            if len(deduped) >= limit:
                break

        return deduped, ranked_all

    def recommend_with_constraint_groups(
        self,
        liked_movie_list: list[str] | None,
        entity_qids: list[str] | None = None,
        limit: int = 5,
        per_group_limit: int = 2,
    ) -> dict:
        """
        Build one candidate set per extracted entity (checkbox).
        If a full intersection exists, return only those movies.
        Otherwise, return grouped recommendations for maximal checkbox groups.
        """
        liked_movie_list = liked_movie_list or []
        entity_qids = entity_qids or []
        preference_profile = self.build_preference_profile(liked_movie_list, entity_qids)
        if not preference_profile:
            return {"mode": "empty", "profile": preference_profile, "movies": [], "groups": []}

        entity_boxes: list[dict] = []
        for qid in entity_qids:
            movies = self.candidate_movies_for_entity_qid(qid)
            if movies:
                entity_boxes.append({"qid": qid, "movies": movies})

        if not entity_boxes:
            ranked = self._rank_candidate_pool(set(self.m_attrs.keys()), preference_profile, liked_movie_list, limit)
            return {
                "mode": "scored",
                "profile": preference_profile,
                "movies": [mid for mid, _ in ranked],
                "ranked": ranked,
                "groups": [],
            }

        full_candidates = set(entity_boxes[0]["movies"])
        for box in entity_boxes[1:]:
            full_candidates &= set(box["movies"])

        if full_candidates:
            if not liked_movie_list:
                selected_movies, ranked = self._select_entity_only_full_intersection_movies(
                    full_candidates,
                    preference_profile,
                    liked_movie_list,
                    limit,
                )
                return {
                    "mode": "all",
                    "profile": preference_profile,
                    "movies": selected_movies,
                    "ranked": ranked,
                    "groups": [],
                    "entity_qids": [box["qid"] for box in entity_boxes],
                }

            ranked = self._rank_candidate_pool(full_candidates, preference_profile, liked_movie_list, limit)
            return {
                "mode": "all",
                "profile": preference_profile,
                "movies": [mid for mid, _ in ranked],
                "ranked": ranked,
                "groups": [],
                "entity_qids": [box["qid"] for box in entity_boxes],
            }

        union_candidates: set[str] = set()
        for box in entity_boxes:
            union_candidates |= set(box["movies"])

        groups: dict[tuple[str, ...], set[str]] = {}
        for movie_id in union_candidates:
            checked = tuple(box["qid"] for box in entity_boxes if movie_id in box["movies"])
            if checked:
                groups.setdefault(checked, set()).add(movie_id)

        if not groups:
            ranked = self._rank_candidate_pool(union_candidates, preference_profile, liked_movie_list, limit)
            return {
                "mode": "scored",
                "profile": preference_profile,
                "movies": [mid for mid, _ in ranked],
                "ranked": ranked,
                "groups": [],
            }

        all_keys = list(groups.keys())
        maximal_keys: list[tuple[str, ...]] = []
        for key in sorted(all_keys, key=lambda k: (-len(k), list(k))):
            key_set = set(key)
            if any(key_set < set(other) for other in all_keys if other != key):
                continue
            maximal_keys.append(key)

        group_payload = []
        for key in sorted(maximal_keys, key=lambda k: (-len(k), list(k))):
            ranked = self._rank_candidate_pool(groups[key], preference_profile, liked_movie_list, max(per_group_limit, len(groups[key])))
            if not liked_movie_list:
                selected_movies = self._sample_random_from_highest_tiers(groups[key], per_group_limit)
            else:
                if not ranked:
                    continue
                selected_movies = [mid for mid, _ in ranked[:per_group_limit]]
            if not selected_movies:
                continue
            group_payload.append({
                "entity_qids": list(key),
                "movies": selected_movies,
                "ranked": ranked,
            })

        if not group_payload:
            ranked = self._rank_candidate_pool(union_candidates, preference_profile, liked_movie_list, limit)
            return {
                "mode": "scored",
                "profile": preference_profile,
                "movies": [mid for mid, _ in ranked],
                "ranked": ranked,
                "groups": [],
            }

        return {
            "mode": "groups",
            "profile": preference_profile,
            "movies": [],
            "ranked": [],
            "groups": group_payload,
            "all_entity_qids": [box["qid"] for box in entity_boxes],
        }


    def recommend(
        self,
        liked_movie_list: list[str] | None,
        entity_qids: list[str] | None = None,
        limit: int = 5,
    ):
        """
        Default movie recommender.
        If entities are present, first try the checkbox/group-aware recommendation logic.
        Otherwise keep the existing content-based scoring + KNN refinement behavior.
        """
        liked_movie_list = liked_movie_list or []
        entity_qids = entity_qids or []

        if entity_qids:
            structured = self.recommend_with_constraint_groups(
                liked_movie_list,
                entity_qids,
                limit=limit,
                per_group_limit=2,
            )
            if structured["mode"] == "all":
                return structured["movies"]
            if structured["mode"] == "groups":
                flattened = []
                seen = set()
                for group in structured["groups"]:
                    for mid in group.get("movies", []):
                        if mid not in seen:
                            seen.add(mid)
                            flattened.append(mid)
                if flattened:
                    return flattened[:limit]
            if structured["mode"] == "scored":
                return structured.get("movies", [])

        preference_profile = self.build_preference_profile(liked_movie_list, entity_qids)

        # if we ended up with an empty profile, just bail
        if not preference_profile:
            return []

        # ---- 1) Content-based scoring ----
        recommendation: dict[str, float] = {}
        for movie_id, attrs in self.m_attrs.items():
            score = 0.0

            for group, values in attrs.items():
                # normalize values to an iterable
                if not isinstance(values, (set, list, tuple)):
                    values = [values]

                for v in values:
                    score += preference_profile.get((group, v), 0.0)

            # skip movies with zero score and the ones the user already likes
            if score > 0 and movie_id not in liked_movie_list:
                recommendation[movie_id] = score

        # no matches at all → just bail
        if not recommendation:
            return []

        # ---- 2) Top-N content-based ----
        sorted_recommendation = sorted(
            recommendation.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        top_movies = sorted_recommendation[:limit]
        top_recommendation = dict(top_movies)   # {movie_id: score}

        # ---- 3) Try KNN refinement on these candidates ----
        top_movie_ids = list(top_recommendation.keys())
        knn = KNNRecommender(self.r_m, content_recommendations=top_movie_ids)

        knn_recommendation = knn.recommend(liked_movie_list, top_k=limit)

        # ---- 4) Fallback if KNN gives nothing ----
        if knn_recommendation:
            return knn_recommendation
        else:
            return top_movie_ids

    def explain_movie_against_profile(self, movie_id: str, preference_profile: dict[tuple[str, str], float]) -> dict:
        attrs = self.m_attrs.get(movie_id, {}) or {}
        matched_attributes: list[dict] = []
        total_score = 0.0

        for group, values in attrs.items():
            if not isinstance(values, (set, list, tuple)):
                values = [values]
            for value in values:
                contribution = float(preference_profile.get((group, value), 0.0))
                if contribution <= 0:
                    continue
                total_score += contribution
                matched_attributes.append({
                    "group": group,
                    "value": value,
                    "score": contribution,
                })

        matched_attributes.sort(key=lambda item: (-item["score"], str(item["group"]), str(item["value"])))
        return {
            "movie_id": movie_id,
            "total_score": total_score,
            "match_count": len(matched_attributes),
            "matched_attributes": matched_attributes,
        }

    def debug_liked_movie_profile_rows(
        self,
        liked_movie_list: list[str] | None,
    ) -> list[dict]:
        """
        Debug helper for title-based bag scoring.
        Returns one row per (attr, value) built from liked movies only,
        including exact final score, base weight, multiplier, and overlap count.
        """
        liked_movie_list = liked_movie_list or []
        if not liked_movie_list:
            return []

        preference_profile = self.build_preference_profile(liked_movie_list, [])
        total_liked_movies = len(liked_movie_list)
        rows: list[dict] = []

        for (attr, value), final_score in preference_profile.items():
            movie_ids_with_attr: list[str] = []
            for movie_id in liked_movie_list:
                movie_attrs = self.m_attrs.get(movie_id, {}) or {}
                values = movie_attrs.get(attr)
                if values and value in values:
                    movie_ids_with_attr.append(movie_id)

            overlap_count = len(movie_ids_with_attr)
            base_weight = float(self._profile_weight_for_attr_value(attr, value, weighted_title_mode=True))
            if overlap_count > 1:
                multiplier = float(5 ** (overlap_count - 1))
                multiplier_reason = "overlap_boost"
            else:
                multiplier = float(0.8 ** total_liked_movies if total_liked_movies > 0 else 1.0)
                multiplier_reason = "singleton_penalty"

            rows.append({
                "group": attr,
                "value": value,
                "score": float(final_score),
                "base_weight": base_weight,
                "multiplier": multiplier,
                "multiplier_reason": multiplier_reason,
                "overlap_count": overlap_count,
                "source_movie_ids": list(movie_ids_with_attr),
            })

        rows.sort(key=lambda item: (-float(item["score"]), str(item["group"]), str(item["value"])))
        return rows

    def rank_movies_by_profile(
        self,
        liked_movie_list: list[str] | None,
        entity_qids: list[str] | None = None,
    ) -> tuple[dict[tuple[str, str], float], list[tuple[str, float]]]:
        preference_profile = self.build_preference_profile(liked_movie_list, entity_qids)
        if not preference_profile:
            return preference_profile, []

        recommendation: dict[str, float] = {}
        liked_set = set(liked_movie_list or [])
        for movie_id in self.m_attrs.keys():
            if movie_id in liked_set:
                continue
            explanation = self.explain_movie_against_profile(movie_id, preference_profile)
            if explanation["total_score"] > 0:
                recommendation[movie_id] = float(explanation["total_score"])

        sorted_recommendation = sorted(
            recommendation.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        return preference_profile, sorted_recommendation

    def recommend_for_entities(self, entity_qids: list[str], limit: int = 5) -> list[str]:
        """
        Recommend movies that are linked to the given entities (actors, genres, etc.).
        entity_qids are bare Q-IDs ('Q38142', 'Q33968', ...).
        """
        if not self.attr_index or not entity_qids:
            return []

        candidate_sets: list[set[str]] = []

        # For each entity QID, find all movies that have it in *any* attribute.
        for qid in entity_qids:
            norm_qid = self._normalize_qid(qid)
            movies_for_qid: set[str] = set()

            for attr, value_to_movies in self.attr_index.items():
                # 1) Try direct lookup with bare QID
                movies = value_to_movies.get(norm_qid)
                if movies:
                    movies_for_qid |= set(movies)
                    continue

                # 2) Fallback: keys may be full IRIs, normalize them
                for raw_value, raw_movies in value_to_movies.items():
                    if self._normalize_qid(str(raw_value)) == norm_qid:
                        movies_for_qid |= set(raw_movies)

            if movies_for_qid:
                candidate_sets.append(movies_for_qid)

        if not candidate_sets:
            return []

        # AND semantics: movies that satisfy *all* constraints
        candidates = candidate_sets[0]
        for s in candidate_sets[1:]:
            candidates &= s

        # If intersection is empty, fall back to OR (any of the constraints)
        if not candidates:
            candidates = set().union(*candidate_sets)

        # Rank candidates by popularity / mean rating from rating_matrix
        scores: dict[str, float] = {}
        for mid in candidates:
            if mid not in self.r_m.index:
                continue
            row = self.r_m.loc[mid]
            nonzero = row[row != 0]
            if len(nonzero) == 0:
                continue
            scores[mid] = float(nonzero.mean())

        if not scores:
            return []

        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [mid for mid, _ in top]



class ContentRecommenderDebugger:
    def __init__(self, cr: ContentRecommender):
        self.cr = cr

    def debug(self):
        liked_movie_list = [
            "http://www.wikidata.org/entity/Q927048",
            "http://www.wikidata.org/entity/Q36479",
            "http://www.wikidata.org/entity/Q218894",
            ]
        with debug_block():
            print("Content based recommendations:\n")
            for qid, score in self.cr.recommend(liked_movie_list).items():
                print(f"{qid}: {score}")
