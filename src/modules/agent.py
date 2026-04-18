import time
from pathlib import Path
import os
import random
import re
from typing import Iterable

from modules.config import Config
from modules.logger import Logger
from modules.intent_identifier import IntentIdentifier
from modules.prefix_extractor import PrefixExtractor
from modules.approach_identifier import ApproachIdentifier, Approach
from utils.message_blocks import debug_block
from modules.recommender.recommender_file_helper import RatingsFileProcessor
from modules.recommender.matrix_builder import RatingMatrixBuilder
from modules.recommender.recommender import KNNRecommender, ContentRecommender
from modules.recommender.return_recommendation import Recommendation
from modules.title_resolver import TitleResolver
from modules.runtime.artifact_store import RuntimeArtifactStore
from modules.dict_entity_linker import DictEntityLinker
from modules.extractors import Extractor
from modules.entity_synonym_matcher import EntitySynonymMatcher
from modules.paths import ensure_datapath
from modules.popularity_tier import POPULARITY_TIER_RELATION

ATAI_TAG = "http://ddis.ch/atai/tag"

REPORT_MAIN_RELATIONS = {
    "http://www.wikidata.org/prop/direct/P57",   # director
    "http://www.wikidata.org/prop/direct/P144",  # based on
    "http://www.wikidata.org/prop/direct/P364",  # original language
    "http://www.wikidata.org/prop/direct/P495",  # country of origin
}
REPORT_GENRE_TAG_RELATIONS = {
    ATAI_TAG,
    "http://www.wikidata.org/prop/direct/P136",  # genre
}
REPORT_CREW_RELATIONS = {
    "http://www.wikidata.org/prop/direct/P1040",  # film editor
    "http://www.wikidata.org/prop/direct/P162",   # producer
    "http://www.wikidata.org/prop/direct/P58",    # screenwriter
    "http://www.wikidata.org/prop/direct/P86",    # composer
    "http://www.wikidata.org/prop/direct/P2554",  # production designer
}
REPORT_AWARD_RELATIONS = {
    "http://www.wikidata.org/prop/direct/P1411",  # nominated for
    "http://www.wikidata.org/prop/direct/P166",   # award received
}
REPORT_CAST_RELATIONS = {
    "http://www.wikidata.org/prop/direct/P161",   # cast member
}


class Agent:
    """
    Graph-free runtime agent.

    Uses only:
      - prebuilt runtime artifacts (movie_attrs, attr_index, label dictionaries)
      - titles_to_qid.json
      - ratings artifacts

    The RDF graph is no longer loaded or queried at runtime.
    """

    def __init__(self, config: Config):
        self.logger = Logger()
        self.config = config

        self.data_root = ensure_datapath()
        self.artifacts_dir = Path(getattr(config, "artifacts_dir", self.data_root / "runtime_artifacts"))

        missing = RuntimeArtifactStore.validate(self.artifacts_dir)
        if missing:
            raise FileNotFoundError(
                f"Missing runtime artifacts in {self.artifacts_dir}: {', '.join(missing)}. "
                f"Run build_runtime_artifacts.py first."
            )

        self.intent_identifier = IntentIdentifier()
        self.approach_identifier = ApproachIdentifier()
        self.title_resolver = TitleResolver()

        self.rp = RatingsFileProcessor(self.config.ratings_path)
        self.rmb = RatingMatrixBuilder(self.rp)
        self.rating_matrix = None

        self.movie_attrs: dict[str, dict[str, set[str]]] = {}
        self.attr_index: dict[str, dict[str, set[str]]] = {}
        self.id_relation_dict: dict[str, str] = {}
        self.relation_id_dict: dict[str, str] = {}
        self.id_entity_dict: dict[str, str] = {}
        self.entity_id_dict: dict[str, str] = {}
        self.id_movie_dict: dict[str, str] = {}
        self.movie_id_dict: dict[str, str] = {}

        self.knnrec = None
        self.conrec = None

        self.setup()

    def setup(self):
        self.rp.recommender_file_prep()
        self.rmb.build_rating_matrix()
        self.rating_matrix = self.rmb.rating_matrix

        store = RuntimeArtifactStore(self.artifacts_dir)
        self.movie_attrs = store.movie_attrs
        self.attr_index = store.attr_index
        self.id_movie_dict = store.id_movie_dict
        self.movie_id_dict = store.movie_id_dict
        self.id_relation_dict = store.id_relation_dict
        self.relation_id_dict = store.relation_id_dict
        self.id_entity_dict = store.id_entity_dict
        self.entity_id_dict = store.entity_id_dict
        self.id_relation_dict.setdefault(POPULARITY_TIER_RELATION, "popularity tier")
        self.relation_id_dict.setdefault("popularity tier", POPULARITY_TIER_RELATION)

        self.entity_synonym_matcher = EntitySynonymMatcher(
            self.entity_id_dict,
            self.artifacts_dir / "entity_synonyms.json",
        )
        self.entity_linker = DictEntityLinker(
            self.movie_id_dict,
            self.entity_id_dict,
            self.entity_synonym_matcher,
        )
        self.extractor = Extractor(
            self.id_relation_dict,
            self.entity_id_dict,
            self.id_movie_dict,
            self.entity_synonym_matcher,
        )

        self.knnrec = KNNRecommender(self.rating_matrix)
        self.conrec = ContentRecommender(
            self.movie_attrs,
            self.rating_matrix,
            self.attr_index,
            store.genre_tag_bag_index,
            self.id_entity_dict,
        )
        self.knn_recommendation = Recommendation(self.title_resolver, self.knnrec)
        self.con_recommendation = Recommendation(self.title_resolver, self.conrec)

    def handle_message(self, message: str) -> str:
        print(f"New message: {message}")
        self.logger.log_message(message)

        approach, explicit = self.approach_identifier.identify_approach(message)
        intent_spec = self.intent_identifier.question_to_intent(message, self.entity_linker)

        with debug_block():
            print(
                "[Factual] intent_spec:\n"
                f"  intent={intent_spec.intent}\n"
                f"  entity_label={intent_spec.entity_label}\n"
                f"  entity_iri={intent_spec.entity_iri}\n"
                f"  relation_iri={intent_spec.relation_iri}\n"
                f"  kb_lookup_successful={intent_spec.kb_lookup_successful}\n"
                f"  bad_match={intent_spec.bad_match}\n"
                f"  lookup_result={intent_spec.lookup_result}"
            )


        if intent_spec.intent not in {None, "raw"}:
            if not intent_spec.kb_lookup_successful or (
                intent_spec.entity_label and not intent_spec.entity_iri and intent_spec.intent != "movies_by_tag"
            ):
                if intent_spec.entity_label:
                    return self._final_reply(
                        f"I recognized this as a factual question, but I couldn't resolve "
                        f"{self._format_entity_display(intent_spec.entity_label)} in the local dataset."
                    )
                return self._final_reply("I recognized the factual intent, but I couldn't resolve the target entity.")

            rows = self._lookup_factual(
                intent_spec.intent,
                intent_spec.entity_iri,
                intent_spec.relation_iri,
                intent_spec.entity_label,
            )
            if not rows:
                return self._final_reply("I couldn't find a factual answer for that in the local dataset.")

            body, shown = self._format_results(intent_spec.intent, rows)
            if (
                intent_spec.bad_match
                and intent_spec.lookup_result
                and intent_spec.lookup_result != intent_spec.entity_label
            ):
                body = f"I matched your query to {self._format_entity_display(intent_spec.lookup_result)}.\n" + body
            return self._final_reply(body)

        return self._run_recommendation_logic(message)

    def on_new_message(self, message, room):
        response = self.handle_message(message)
        if hasattr(room, "post_messages"):
            room.post_messages(response)
        return response

    def _final_reply(self, text: str) -> str:
        self.logger.log_response(text)
        return text

    def _capitalize_each_word(self, text: str) -> str:
        return " ".join(token[:1].upper() + token[1:] if token else token for token in str(text).split())

    def _format_movie_display(self, title: str) -> str:
        return f"**{self._capitalize_each_word(title)}**"

    def _format_entity_display(self, label: str) -> str:
        return f"**{self._capitalize_each_word(label)}**"

    def _format_movie_list(self, titles: list[str]) -> list[str]:
        return [self._format_movie_display(title) for title in titles]

    def _format_entity_list(self, labels: list[str]) -> list[str]:
        return [self._format_entity_display(label) for label in labels]

    def _entity_label_from_qid(self, qid: str, fallback: str | None = None) -> str:
        if qid in self.id_entity_dict:
            return self.id_entity_dict[qid]
        qid_tail = qid.rstrip('/').split('/')[-1] if isinstance(qid, str) else str(qid)
        for iri, label in self.id_entity_dict.items():
            if iri.rstrip('/').split('/')[-1] == qid_tail:
                return label
        return fallback or qid_tail

    def _is_release_year_entity_label(self, label: str) -> bool:
        value = str(label or "").strip()
        return bool(re.fullmatch(r"(?:19|20)\d{2}", value))

    def _recommendation_attr_ids_from_entities(self, entities: list[str]) -> list[str]:
        attr_ids: list[str] = []
        for ent in entities or []:
            if self._is_release_year_entity_label(ent):
                attr_ids.append(f"{self.conrec.RELEASE_YEAR_ENTITY_PREFIX}{ent}")
            elif ent in self.entity_id_dict:
                attr_ids.append(self.entity_id_dict[ent])
        return attr_ids

    def _format_profile_value_display(self, attr: str, value: str) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        if attr == self.conrec.RELEASE_DATE_RELATION:
            year = self.conrec._release_year_from_raw_value(raw)
            if year:
                return year
        return self._entity_label_from_qid(raw, fallback=raw)

    def _movie_display_from_qid(self, movie_qid: str) -> str:
        title = self.title_resolver.title_from_qid([movie_qid])
        if title:
            return self._format_movie_display(title[0])
        fallback = self.id_movie_dict.get(movie_qid, movie_qid)
        return self._format_movie_display(fallback)

    def _movie_display_list_from_qids(self, movie_qids: list[str]) -> list[str]:
        return [self._movie_display_from_qid(mid) for mid in movie_qids]

    def _join_bold_items(self, items: list[str]) -> str:
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return items[0] + " and " + items[1]
        return ", ".join(items[:-1]) + ", and " + items[-1]

    def _report_category_key(self, relation_iri: str) -> str | None:
        if relation_iri in REPORT_MAIN_RELATIONS:
            return "main"
        if relation_iri in REPORT_GENRE_TAG_RELATIONS:
            return "genre_tag"
        if relation_iri in REPORT_CREW_RELATIONS:
            return "crew"
        if relation_iri in REPORT_AWARD_RELATIONS:
            return "award"
        if relation_iri in REPORT_CAST_RELATIONS:
            return "cast"
        return None

    def _relation_display_name(self, relation_iri: str) -> str:
        return self.id_relation_dict.get(relation_iri, relation_iri)

    def _format_report_line(self, row: dict) -> str:
        relation_iri = str(row.get("group") or "")
        value = str(row.get("value") or "")
        value_display = self._format_entity_display(self._format_profile_value_display(relation_iri, value))
        movie_ids = list(row.get("source_movie_ids") or [])
        movie_displays = self._movie_display_list_from_qids(movie_ids)
        movie_phrase = self._join_bold_items(movie_displays)
        count = len(movie_ids)
        relation_name = self._relation_display_name(relation_iri).lower()

        if relation_iri in REPORT_GENRE_TAG_RELATIONS:
            kind = "tag" if relation_iri == ATAI_TAG else "genre"
            return f"Your liked films commonly share {kind} {value_display} — {movie_phrase}."

        if relation_iri == "http://www.wikidata.org/prop/direct/P1411":
            return f"{count} of them — {movie_phrase} — were nominated for {value_display}."

        if relation_iri == "http://www.wikidata.org/prop/direct/P166":
            return f"{count} of them — {movie_phrase} — received {value_display}."

        if relation_iri == "http://www.wikidata.org/prop/direct/P144":
            return f"{count} of them — {movie_phrase} — are based on {value_display}."
        if relation_iri == "http://www.wikidata.org/prop/direct/P364":
            return f"{count} of them — {movie_phrase} — have the same original language — {value_display}."
        if relation_iri == "http://www.wikidata.org/prop/direct/P495":
            return f"{count} of them — {movie_phrase} — have the same country of origin — {value_display}."
        return f"{count} of them — {movie_phrase} — have the same {relation_name} — {value_display}."

    def _multi_movie_profile_report(self, liked_qids_full: list[str]) -> str:
        if len(liked_qids_full) <= 1:
            return ""

        rows = self.conrec.debug_liked_movie_profile_rows(liked_qids_full)
        eligible_rows = [row for row in rows if float(row.get("score", 0.0)) >= 5.0]
        if not eligible_rows:
            return ""

        buckets: dict[str, list[dict]] = {"main": [], "genre_tag": [], "crew": [], "award": [], "cast": []}
        for row in eligible_rows:
            key = self._report_category_key(str(row.get("group") or ""))
            if key is None:
                continue
            buckets[key].append(row)

        category_limits = {
            "main": 1,
            "genre_tag": 2,
            "crew": 2,
            "award": 1,
            "cast": 2,
        }
        available_categories = [key for key, bucket_rows in buckets.items() if bucket_rows]
        if not available_categories:
            return ""

        rng = random.SystemRandom()
        chosen_categories = rng.sample(available_categories, min(2, len(available_categories)))
        lines: list[str] = []
        for category in chosen_categories:
            bucket_rows = list(buckets.get(category, []))
            limit = int(category_limits.get(category, 0))
            if not bucket_rows or limit <= 0:
                continue
            rng.shuffle(bucket_rows)
            selected_rows = bucket_rows[:limit]
            for row in selected_rows:
                lines.append(self._format_report_line(row))

        if not lines:
            return ""
        return "\n\n" + "\n".join(lines)

    def _format_grouped_recommendation_reply(

        self,
        prefix: str,
        structured: dict,
        entity_display_map: dict[str, str],
        strongest_profile_report: str = "",
    ) -> str:
        mode = structured.get("mode")
        if mode == "all":
            rec_qids = structured.get("movies", [])
            rec_titles = self.title_resolver.title_from_qid(rec_qids)
            if not rec_titles:
                return self._final_reply("I found matching movies, but couldn't map them back to titles.")
            reply = prefix + "I recommend " + ", ".join(self._format_movie_list(rec_titles)) + "." + strongest_profile_report
            return self._final_reply(reply)

        if mode == "groups":
            lines = [prefix.rstrip(), "I couldn't find a movie that contains all of the attributes you mentioned."]
            for group in structured.get("groups", []):
                labels = self._format_entity_list([entity_display_map.get(qid, self._entity_label_from_qid(qid)) for qid in group.get("entity_qids", [])])
                movie_titles = self.title_resolver.title_from_qid(group.get("movies", []))
                if not movie_titles:
                    continue
                if len(labels) == 1:
                    group_phrase = labels[0]
                elif len(labels) == 2:
                    group_phrase = labels[0] + " and " + labels[1]
                else:
                    group_phrase = ", ".join(labels[:-1]) + ", and " + labels[-1]
                lines.append(f"Here are films that match {group_phrase}: " + ", ".join(self._format_movie_list(movie_titles)) + ".")
            return self._final_reply("\n".join(line for line in lines if line) + strongest_profile_report)

        if mode == "scored":
            rec_qids = structured.get("movies", [])
            rec_titles = self.title_resolver.title_from_qid(rec_qids)
            if rec_titles:
                return self._final_reply(prefix + "I recommend " + ", ".join(self._format_movie_list(rec_titles)) + "." + strongest_profile_report)
            return self._final_reply("I couldn't find any recommendations for that.")

        return self._final_reply("I couldn't find any recommendations for that.")

    def on_new_reaction(self, reaction: str, message_ordinal: int, room=None):
        print(f"New reaction '{reaction}' on message #{message_ordinal}")
        response = f"Thanks for your reaction: '{reaction}'"
        self.logger.log_reaction(reaction, message_ordinal)
        self.logger.log_response(response)
        return response

    def _run_recommendation_logic(self, message: str):
        prefix = "Given you like "
        extracted = self.extractor.extract_rec_mulm(message)
        if extracted is None:
            return self._final_reply(
                "I couldn't extract any known movie titles, actors, or genres in your request. "
                "Try something like: 'Recommend movies like Inception' or 'Recommend biographical films with Meryl Streep'."
            )

        titles = list(extracted.get("titles", []))
        entities = list(extracted.get("entities", []))
        clarify_titles = list(extracted.get("clarify_titles", []))
        liked_qids_full = self.title_resolver.qid_from_title(titles)
        attr_qids = self._recommendation_attr_ids_from_entities(entities)

        with debug_block():
            print(f"[Rec] Extracted titles: {titles} -> {liked_qids_full}")
            print(f"[Rec] Extracted attribute entities: {entities} -> {attr_qids}")

        if titles:
            prefix += "movies like "
        for i, title in enumerate(titles):
            display_title = self._format_movie_display(title)
            if i < 2:
                if i > 0:
                    prefix += ', ' + display_title
                else:
                    prefix += display_title
            if i > 1:
                prefix += "..."
        if titles and entities:
            prefix += ", and your preference: "
        for i, ent in enumerate(entities):
            display_entity = self._format_entity_display(ent)
            if i < 2:
                if i > 0:
                    prefix += ', ' + display_entity
                else:
                    prefix += display_entity
            if i > 1:
                prefix += "..."
        prefix += "\n"
        strongest_profile_report = self._multi_movie_profile_report(liked_qids_full)

        if not liked_qids_full and not attr_qids:
            if clarify_titles:
                example = clarify_titles[0]
                display_example = self._format_movie_display(example)
                reply = (
                    f"I couldn't confidently tell whether {display_example} was meant as a movie title. "
                    f"If you meant the movie {display_example}, please say something like \"recommend more movies like {self._capitalize_each_word(example)}\"."
                )
                return self._final_reply(reply)
            reply = "I couldn't recognize any movie titles, actors, or genres in your request."
            return self._final_reply(reply)

        entity_display_map = {}
        for ent in entities:
            if self._is_release_year_entity_label(ent):
                qid = f"{self.conrec.RELEASE_YEAR_ENTITY_PREFIX}{ent}"
            elif ent in self.entity_id_dict:
                qid = self.entity_id_dict.get(ent)
            else:
                qid = None
            if qid:
                entity_display_map.setdefault(qid, ent)

        if attr_qids:
            structured = self.conrec.recommend_with_constraint_groups(
                liked_qids_full,
                attr_qids,
                limit=5,
                per_group_limit=2,
            )
            if structured.get("mode") in {"all", "groups", "scored"}:
                # If nothing useful came back in scored mode, keep falling through.
                if structured.get("mode") != "scored" or structured.get("movies"):
                    return self._format_grouped_recommendation_reply(prefix, structured, entity_display_map, strongest_profile_report)

        rec_qids = self.conrec.recommend(liked_qids_full, attr_qids)
        if not rec_qids:
            reply = "I couldn't find any recommendations for that."
            return self._final_reply(reply)

        rec_titles = self.title_resolver.title_from_qid(rec_qids)
        if not rec_titles:
            reply = "I found some matching movies, but couldn't map them back to titles."
            return self._final_reply(reply)

        reply = "I recommend " + ", ".join(self._format_movie_list(rec_titles)) + "."
        reply = prefix + reply + strongest_profile_report
        return self._final_reply(reply)

    def _lookup_factual(self, intent: str, entity_iri: str, relation_iri: str, entity_label: str | None) -> list[str]:
        if not entity_iri:
            return []

        # inverse movie lookup requests
        if intent in {"movies_by_director", "movies_by_writer", "movies_by_actor", "movies_by_producer", "movies_by_composer", "movies_by_editor", "movies_by_cinematographer"}:
            movie_ids = self.attr_index.get(relation_iri, {}).get(entity_iri, set())
            return self._labels_for_iris(sorted(movie_ids))

        if intent == "movies_by_tag":
            tag_map = self.attr_index.get(ATAI_TAG, {})
            if not entity_label:
                return []
            movies = tag_map.get(entity_label.lower(), set()) or tag_map.get(entity_label, set())
            return self._labels_for_iris(sorted(movies))

        values = list(self.movie_attrs.get(entity_iri, {}).get(relation_iri, set()))
        if not values:
            return []

        if intent == "release_year":
            return [str(v)[:4] for v in values if v]

        # convert IRIs to labels when possible; otherwise return literals directly
        rendered = []
        for v in values:
            label = self._label_for_iri(v)
            rendered.append(label if label is not None else str(v))
        return rendered

    def _label_for_iri(self, iri: str) -> str | None:
        if iri in self.id_movie_dict:
            return self.id_movie_dict[iri]
        if iri in self.id_entity_dict:
            return self.id_entity_dict[iri]
        return None

    def _labels_for_iris(self, iris: Iterable[str]) -> list[str]:
        labels: list[str] = []
        for iri in iris:
            lbl = self._label_for_iri(iri)
            labels.append(lbl if lbl is not None else str(iri))
        return labels

    def _format_results(self, intent: str, rows: list[str]) -> str:
        if not rows:
            return "No results."

        dedup = []
        for v in rows:
            if v is None or v == "None":
                continue
            dedup.append(v)
        dedup = list(dict.fromkeys(dedup))

        if not dedup:
            return "No results."

        MAX_DISPLAY = 20
        shown = dedup[:MAX_DISPLAY]
        leftover = max(0, len(dedup) - MAX_DISPLAY)
        body = PrefixExtractor().prefix_for(intent)
        with debug_block():
            print(f"intent is: {intent}" f"prefix extracted is: {body}")
        if intent in {"release_date", "release_year", "box_office", "custom_rating"}:
            return body + shown[0], shown[0]
        shown = " and ".join(shown)
        if leftover:
            shown += f"\n… (+{leftover} more, truncated)"
        body += shown
        return body, shown

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())


class MovieFunFactHelper:
    FUN_FACTS = [
        "The sound of E.T. walking was made by someone squishing their hands in jelly!",
        "The code in The Matrix comes from sushi recipes!",
        "In The Wizard of Oz, Toto was paid more than the Munchkins!",
        "The iconic roar of the MGM lion is actually a tiger's roar!",
        "Pixar's original name was 'The Graphics Group'.",
        "The first film with a plot was 'The Great Train Robbery' in 1903, and it was only 12 minutes long!",
        "Alfred Hitchcock never won an Oscar for Best Director despite 5 nominations!",
        "The average Hollywood movie ticket in 1910 cost 7 cents!",
        "The longest movie ever made is 'The Cure for Insomnia' at 87 hours!",
        "Sean Connery wore a toupee in every James Bond movie he appeared in!",
    ]

    @staticmethod
    def get_random_fact() -> str:
        return random.choice(MovieFunFactHelper.FUN_FACTS)
