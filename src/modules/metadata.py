from dataclasses import dataclass, asdict
from typing import Literal, Optional

WDT = "http://www.wikidata.org/prop/direct/"

@dataclass
class Metadata:
    answer_type: Optional[str] = None
    answer_method: Literal["Factual", "Embedding"] = "Factual"
    answer: str = "None"

    @property
    def message(self) -> str:
        answer_approach_string = f'[Approach] {self.answer_method}\n'
        answer_type_string = f'[Answer Type] {self.answer_type}\n'
        answer = f"[Answer] {self.answer}"
        return f"\n\n[Metadata]\n{answer_type_string+answer_approach_string+answer if (self.answer_method == 'Embedding') else answer_approach_string+answer}"
    
class MetadataHelper:
    def __init__(self, relation_iri: str, use_embedding: bool, answer: str):
        self.relation_iri = relation_iri
        self.use_embedding = use_embedding
        self.answer = answer
        self.relation_entity_mappings = {
            f"{WDT}P57": "director",
            f"{WDT}P58": "screenwriter",
            f"{WDT}P161": "cast member",
            f"{WDT}P136": "genre",
            f"{WDT}P364": "original language of film or TV show",
            f"{WDT}P495": "country of origin"
        }
        self.metadata = Metadata(
            answer_type=(self._get_answer_type() if use_embedding else None),
            answer_method=("Embedding" if self.use_embedding else "Factual"),
            answer = self.answer
        )

    def _get_answer_type(self):
        """Return the entity label corresponding to the relation."""
        return self.relation_entity_mappings[self.relation_iri]
