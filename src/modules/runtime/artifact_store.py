from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class RuntimeArtifactStore:
    artifacts_dir: Path

    def __post_init__(self):
        self.artifacts_dir = Path(self.artifacts_dir)
        self.movie_attrs: Dict[str, dict] = self._load_pickle("movie_attrs.pkl")
        self.attr_index: Dict[str, dict] = self._load_pickle("attr_index.pkl")
        self.id_movie_dict: Dict[str, str] = self._load_json("id_movie.json")
        self.movie_id_dict: Dict[str, str] = {str(v): k for k, v in self.id_movie_dict.items() if v is not None}
        self.id_relation_dict: Dict[str, str] = self._load_json("id_relation.json")
        self.relation_id_dict: Dict[str, str] = {str(v): k for k, v in self.id_relation_dict.items() if v is not None}
        self.id_entity_dict: Dict[str, str] = self._load_json("id_entity.json")
        self.entity_id_dict: Dict[str, str] = {str(v): k for k, v in self.id_entity_dict.items() if v is not None}
        self.entity_synonyms = self._load_optional_json("entity_synonyms.json")
        self.genre_tag_bag_index = self._load_optional_json("genre_tag_bag_index.json")

    def _path(self, name: str) -> Path:
        return self.artifacts_dir / name

    def _load_json(self, name: str) -> Dict[str, Any]:
        p = self._path(name)
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_pickle(self, name: str) -> Any:
        p = self._path(name)
        with p.open("rb") as f:
            return pickle.load(f)

    def _load_optional_json(self, name: str) -> Dict[str, Any] | None:
        p = self._path(name)
        if not p.exists():
            return None
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    def required_files(self) -> list[str]:
        return [
            "movie_attrs.pkl",
            "attr_index.pkl",
            "id_movie.json",
            "id_relation.json",
            "id_entity.json",
        ]

    @classmethod
    def validate(cls, artifacts_dir: Path) -> list[str]:
        required = [
            "movie_attrs.pkl",
            "attr_index.pkl",
            "id_movie.json",
            "id_relation.json",
            "id_entity.json",
        ]
        missing = [name for name in required if not (Path(artifacts_dir) / name).exists()]
        return missing
