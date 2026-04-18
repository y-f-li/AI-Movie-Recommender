from utils.message_blocks import debug_block, reply_block
import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Optional, Any
from modules.pattern_library import PatternLibrary

WDT = "http://www.wikidata.org/prop/direct/"
WD  = "http://www.wikidata.org/entity/"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
TAG = "http://ddis.ch/atai/tag"



@dataclass(frozen=True)
class IntentSpec:
    """
    Specifications for an Intent, including 
    the potential iri of the head entity and relation,
    whether the expected answer is a literal or a tail entity,
    if embedding is appropriate if factual retrieval fails,
    the intent, and a placeholder called entity_label
    for some texts that will help with building the answer string. 
    """
    entity_iri: str | None = None
    relation_iri: str | None = None
    expected_answer_kind: str = "entity"      # "entity" | "literal"
    embedding_ok: bool = True
    intent: str | None = None
    entity_label: str | None = None
    kb_lookup_successful: bool = True
    bad_match: bool = False
    lookup_result: str = None

class IntentIdentifier:
    """
    Regex_driven NL -> IntentSpec for common movie questions.  
    """
    def __init__(self):
        self.library = PatternLibrary()
        self.patterns = self.library.patterns

        self.intent_spec_mappings = {
            "director_of":  IntentSpec(relation_iri=f"{WDT}P57", intent="director_of"),
            "writer_of":    IntentSpec(relation_iri=f"{WDT}P58", intent="writer_of"),

            "release_date": IntentSpec(relation_iri=f"{WDT}P577", embedding_ok=False, intent="release_date"),
            "release_year": IntentSpec(relation_iri=f"{WDT}P577", embedding_ok=False, intent="release_year"),

            "cast_of":  IntentSpec(relation_iri=f"{WDT}P161", intent="cast_of"),

            "movies_by_director":   IntentSpec(relation_iri=f"{WDT}P57", intent="movies_by_director"),
            "movies_by_writer": IntentSpec(relation_iri=f"{WDT}P58", intent="movies_by_writer"),
            "movies_by_actor":  IntentSpec(relation_iri=f"{WDT}P161", intent="movies_by_actor"),
            "movies_by_producer": IntentSpec(relation_iri=f"{WDT}P162", intent="movies_by_producer"),
            "movies_by_composer": IntentSpec(relation_iri=f"{WDT}P86", intent="movies_by_composer"),
            "movies_by_editor": IntentSpec(relation_iri=f"{WDT}P1040", intent="movies_by_editor"),
            "movies_by_cinematographer": IntentSpec(relation_iri=f"{WDT}P344", intent="movies_by_cinematographer"),

            "genre_of": IntentSpec(relation_iri=f"{WDT}P136", intent="genre_of"),
            "language_of":  IntentSpec(relation_iri=f"{WDT}P364", intent="language_of"),
            "country_of":   IntentSpec(relation_iri=f"{WDT}P495", intent="country_of"),

            "movies_by_tag":    IntentSpec(intent="movies_by_tag", embedding_ok=False),
            "producer_of":        IntentSpec(relation_iri=f"{WDT}P162", intent="producer_of"),
            "production_company": IntentSpec(relation_iri=f"{WDT}P272", intent="production_company"),
            
            "award_received":     IntentSpec(relation_iri=f"{WDT}P166", intent="award_received"),
            "nominated_for":      IntentSpec(relation_iri=f"{WDT}P1411", intent="nominated_for"),
            
            "box_office":         IntentSpec(relation_iri=f"{WDT}P2142", intent="box_office", expected_answer_kind="literal", embedding_ok=False),
            
            "composer":           IntentSpec(relation_iri=f"{WDT}P86", intent="composer"),
            "cinematographer":    IntentSpec(relation_iri=f"{WDT}P344", intent="cinematographer"),
            "editor":             IntentSpec(relation_iri=f"{WDT}P1040", intent="editor"),
            
            "filming_location":   IntentSpec(relation_iri=f"{WDT}P915", intent="filming_location"),
            "narrative_location": IntentSpec(relation_iri=f"{WDT}P840", intent="narrative_location"),
            
            "based_on":           IntentSpec(relation_iri=f"{WDT}P144", intent="based_on"),
            "mpaa_rating":        IntentSpec(relation_iri=f"{WDT}P1657", intent="mpaa_rating"),
            "main_subject":       IntentSpec(relation_iri=f"{WDT}P921", intent="main_subject"),
            "distributor":        IntentSpec(relation_iri=f"{WDT}P750", intent="distributor"),

            "voice_actor":    IntentSpec(relation_iri=f"{WDT}P725", intent="voice_actor"),
            "characters_in":  IntentSpec(relation_iri=f"{WDT}P674", intent="characters_in"),
            "part_of_series": IntentSpec(relation_iri=f"{WDT}P179", intent="part_of_series"),
            
            "custom_rating":  IntentSpec(relation_iri="http://ddis.ch/atai/rating", intent="custom_rating", expected_answer_kind="literal", embedding_ok=False),
        }
    
    def question_to_intent(self, question: str, entity_linker) -> IntentSpec:
        """
        Retrieve the specifications for an Intent, including 
        the potential iri of the head entity and relation,
        whether the expected answer is a literal or a tail entity,
        if embedding is appropriate if factual retrieval fails,
        the intent, and a placeholder called entity_label
        for some texts that will help with building the answer string. 
        """
        q = question.strip()
        # Normalize Unicode apostrophes and quotes to ASCII
        q = q.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
        for rx, intent in self.patterns:
            m = rx.search(q)
            if not m:
                continue

            entity_label = m.group(1).strip().rstrip('?') if m.groups() else None
            if entity_label:
                entity_label = self.to_kb_punct_label(entity_label)
                entity_label = self._esc_for_sparql(entity_label)
            # DEBUG
            with debug_block():
                print(f"Extracted entity_label: {entity_label}")
            # \DEBUG
            # If caller already resolved entity_iri (movie/person), we use it; else we will resolve outside.
            base = self.intent_spec_mappings[intent]
            intent_specification = replace(base, entity_label=entity_label)

            # If intent is one of the ones that needs an entity, resolve it:
            needs_entity = intent in {
                "director_of","writer_of","release_date","release_year",
                "cast_of","genre_of","language_of","country_of",
                "movies_by_director","movies_by_writer","movies_by_actor",
                "movies_by_producer","movies_by_composer","movies_by_editor","movies_by_cinematographer",
                "producer_of", "production_company", 
                "award_received", "nominated_for", 
                "box_office", 
                "composer", "cinematographer", "editor", 
                "filming_location", "narrative_location", 
                "based_on", "mpaa_rating", "main_subject", "distributor",
                "characters_in",   
                "voice_actor",    
                "part_of_series",  
                "custom_rating"
            }

            if needs_entity and entity_label:
                iri, lookup_result, bad_match = entity_linker.find_entity_by_label(entity_label)
                # DEBUG
                with debug_block():
                    print(f"entity_linker returned iri and lookup result: {iri}, {lookup_result}")
                # /DEBUG                
                intent_specification = replace(intent_specification, entity_iri=iri)
                intent_specification = replace(intent_specification, bad_match=bad_match)
                intent_specification = replace(intent_specification, lookup_result=lookup_result)
            # DEBUG
            with debug_block():
                print("Extracted intent_specification:\n"
                    f"entity_iri: {intent_specification.entity_iri}\n"
                    f"relation_iri: {intent_specification.relation_iri}\n"
                    f"expected_answer_kind: {intent_specification.expected_answer_kind}\n"
                    f"embedding_ok: {intent_specification.embedding_ok}\n"
                    f"intent: {intent_specification.intent}\n"
                    f"Entity Label: {intent_specification.entity_label}\n"
                )
            if needs_entity and not intent_specification.entity_iri:
                with debug_block():
                    print(f"iri is None for detected entity: '{intent_specification.entity_label}'.")
                intent_specification = replace(intent_specification, kb_lookup_successful=False)
                return intent_specification
            # /DEBUG
            return intent_specification
        return IntentSpec(intent="raw", embedding_ok=False)


    def to_kb_punct_label(self, text: str) -> str:
        s = unicodedata.normalize("NFKC", text or "")

        # spaces & edge wrappers
        s = s.replace("\u00A0", " ").replace("\u202F", " ")   # NBSP, NARROW NBSP -> space
        s = s.strip().strip('\'"“”‘’')                        # drop wrapping quotes only

        # colon: KB uses ASCII ':' (map fullwidth to ASCII)
        s = s.replace("：", ":")

        # DASH RULES:
        # 1) If there are spaces around a dash variant, convert that dash to EN DASH.
        #    This covers patterns like "Episode VI - Return" or "VI — Return".
        s = re.sub(r'(?<=\s)[\-–—](?=\s)', "–", s)

        # 2) Optional: if someone typed "VI–Return" (no spaces), add spaces around EN DASH.
        #    Only do this for EN/EM dashes (not hyphens) to avoid touching 'Spider-Man'.
        s = re.sub(r'(?<=\w)[–—](?=\w)', r' – ', s)

        # collapse any weird spacing (and normalize to single spaces)
        s = " ".join(s.split())
        return s

    def _esc_for_sparql(self, s: str) -> str:
        return (s or "").replace("\\", "\\\\").replace('"', '\\"')

