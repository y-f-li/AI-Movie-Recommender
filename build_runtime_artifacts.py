from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

from rdflib import URIRef, RDFS, Literal

# make src importable when run from project root
import sys
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.knowledge_base_loader import KnowledgeBaseLoader
from modules.recommender.recommender_file_helper import RatingsFileProcessor
from modules.recommender.matrix_builder import RatingMatrixBuilder
from modules.entity_helper import extract_wd_id
from build_entity_synonym_dict import build as build_entity_synonym_artifacts
from build_genre_tag_bag_dict import build as build_genre_tag_bag_artifact
from modules.popularity_tier import (
    POPULARITY_TIER_RELATION,
    augment_attr_index_with_popularity,
    augment_movie_attrs_with_popularity,
)

SKIP_LITERAL_ATTRS = {
    "http://schema.org/description",
    str(RDFS.label),
}


def get_en_label(graph, iri: str) -> str | None:
    s = URIRef(iri)
    for lbl in graph.objects(s, RDFS.label):
        if isinstance(lbl, Literal) and (lbl.language == "en" or lbl.language is None):
            return str(lbl)
    return None


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def save_pickle(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def build_runtime_artifacts(dataset_dir: Path):
    graph_path = dataset_dir / "graph.nt"
    ratings_path = dataset_dir / "ratings"
    artifacts_dir = dataset_dir / "runtime_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not graph_path.exists():
        raise FileNotFoundError(f"Missing graph file: {graph_path}")
    if not ratings_path.exists():
        raise FileNotFoundError(f"Missing ratings dir: {ratings_path}")

    os.environ["DATAPATH"] = str(dataset_dir)

    kb_loader = KnowledgeBaseLoader(str(graph_path))
    kb_loader.load()
    graph = kb_loader.graph

    rfp = RatingsFileProcessor(str(ratings_path))
    rfp.recommender_file_prep()
    rmb = RatingMatrixBuilder(rfp)
    rmb.build_rating_matrix()
    rating_matrix = rmb.rating_matrix

    movie_attrs_path = artifacts_dir / "movie_attrs.pkl"
    kb_loader.build_movie_attr_dict(rating_matrix, movie_attrs_path)
    with movie_attrs_path.open("rb") as f:
        movie_attrs = pickle.load(f)

    augment_movie_attrs_with_popularity(movie_attrs)
    save_pickle(movie_attrs_path, movie_attrs)

    attr_index = kb_loader.build_attr_index(movie_attrs)
    augment_attr_index_with_popularity(attr_index, movie_attrs)
    save_pickle(artifacts_dir / "attr_index.pkl", attr_index)

    # id_movie.json
    id_movie = {}
    for movie_iri in movie_attrs.keys():
        label = get_en_label(graph, movie_iri)
        if label:
            id_movie[movie_iri] = label.lower()
    save_json(artifacts_dir / "id_movie.json", id_movie)

    # id_relation.json
    rels = set()
    for attrs in movie_attrs.values():
        rels.update(attrs.keys())
    id_relation = {}
    for rel in rels:
        if rel == POPULARITY_TIER_RELATION:
            id_relation[rel] = "popularity tier"
        elif rel.startswith("http://") or rel.startswith("https://"):
            label = get_en_label(graph, rel)
            id_relation[rel] = (label or rel.rsplit("/", 1)[-1]).lower()
        else:
            id_relation[rel] = str(rel).lower()
    save_json(artifacts_dir / "id_relation.json", id_relation)

    # id_entity.json
    id_entity = {}
    for attr, value_to_movies in attr_index.items():
        for raw_value in value_to_movies.keys():
            if attr in SKIP_LITERAL_ATTRS:
                continue
            val = str(raw_value)
            if val.startswith("http://") or val.startswith("https://"):
                label = get_en_label(graph, val)
                if label:
                    id_entity[val] = label.lower()
                else:
                    tail = extract_wd_id(val)
                    if tail:
                        id_entity[val] = tail.lower()
            else:
                id_entity[val] = val.lower()
    save_json(artifacts_dir / "id_entity.json", id_entity)

    build_entity_synonym_artifacts(dataset_dir)
    build_genre_tag_bag_artifact(dataset_dir)

    print("Wrote runtime artifacts to:", artifacts_dir)
    for name in [
        "movie_attrs.pkl",
        "attr_index.pkl",
        "id_movie.json",
        "id_relation.json",
        "id_entity.json",
        "entity_synonyms.json",
        "word_to_synonyms.json",
        "genre_tag_bag_index.json",
    ]:
        output_path = artifacts_dir / name
        if output_path.exists():
            print(" -", output_path)


if __name__ == "__main__":
    dataset = Path(os.getenv("DATAPATH", ROOT / "dataset")).expanduser().resolve()
    build_runtime_artifacts(dataset)
