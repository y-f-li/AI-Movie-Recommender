from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Set

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.paths import ensure_datapath, get_runtime_artifacts_dir

EXCLUDED_WORDS = {"the", "of", "a", "an", "and", "or"}
CARRIER_WORDS = {"film", "films", "movie", "movies"}
GENRE_RELATION = "http://www.wikidata.org/prop/direct/P136"
ATAI_TAG_RELATION = "http://ddis.ch/atai/tag"

# Small domain bridge seeds to complement generic lexical resources.
TOKEN_SEED_SYNONYMS: Dict[str, Set[str]] = {
    "scary": {"horror", "thriller", "creepy", "frightening", "chilling"},
    "creepy": {"horror", "scary", "chilling", "eerie"},
    "frightening": {"horror", "scary", "terrifying"},
    "dark": {"grim", "morbid", "bleak", "noir", "neo", "twisted"},
    "morbid": {"dark", "grim", "bleak"},
    "grim": {"dark", "bleak", "morbid"},
    "twisted": {"dark", "psychological", "thriller"},
    "biopic": {"biographical", "biographic", "biography"},
    "bio": {"biographical", "biography"},
    "romcom": {"romantic", "comedy"},
    "rom-com": {"romantic", "comedy"},
    "scifi": {"science", "fiction", "sci", "fi"},
    "sci": {"science", "fiction", "scifi", "fi"},
    "fi": {"fiction", "science", "sci", "scifi"},
    "noir": {"dark", "neo", "bleak"},
}

PHRASE_ALIAS_SEEDS: Dict[str, Set[str]] = {
    "biographical film": {"biopic", "bio pic", "biographical movie", "biographical movies"},
    "science fiction film": {"sci fi", "sci-fi", "scifi", "science fiction movie"},
    "romantic comedy": {"romcom", "rom-com", "romantic comedy film", "romantic comedy movie"},
    "road movie": {"road film"},
    "film noir": {"noir film", "dark noir"},
}


def normalize(text: str) -> str:
    s = (text or "").strip().lower()
    s = s.replace("’", "'").replace("‘", "'").replace('“', '"').replace('”', '"')
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", normalize(text)) if t and t not in EXCLUDED_WORDS and t not in CARRIER_WORDS]


def singular_plural_variants(token: str) -> Set[str]:
    t = normalize(token)
    out = {t}
    if not t or len(t) <= 2:
        return out
    if t.endswith("ies") and len(t) > 3:
        out.add(t[:-3] + "y")
    elif t.endswith("es") and len(t) > 3:
        out.add(t[:-2])
    elif t.endswith("s") and not t.endswith("ss") and len(t) > 3:
        out.add(t[:-1])
    else:
        out.add(t + "s")
        out.add(t + "es")
    return {x for x in out if x}


def load_wordnet():
    try:
        import nltk
        from nltk.corpus import wordnet as wn
        try:
            wn.synsets("film")
        except LookupError:
            try:
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
            except Exception:
                pass
        try:
            wn.synsets("film")
            return wn
        except LookupError:
            return None
    except Exception:
        return None


def wordnet_synonyms(token: str, wn, max_items: int = 12) -> Set[str]:
    if wn is None:
        return set()
    results: Set[str] = set()
    try:
        for syn in wn.synsets(token):
            for lemma in syn.lemma_names():
                lemma = normalize(lemma)
                if not lemma or " " in lemma:
                    continue
                if lemma == token:
                    continue
                if not re.fullmatch(r"[a-z0-9]+", lemma):
                    continue
                results.add(lemma)
                if len(results) >= max_items:
                    return results
    except Exception:
        return results
    return results


def build_word_to_synonyms(tokens: Iterable[str], wn) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for token in sorted({normalize(t) for t in tokens if normalize(t)}):
        syns: Set[str] = set()
        syns.update(singular_plural_variants(token))
        syns.update(TOKEN_SEED_SYNONYMS.get(token, set()))
        syns.update(wordnet_synonyms(token, wn))
        syns.add(token)
        out[token] = sorted({normalize(x) for x in syns if normalize(x)})
    return out


def strip_carrier_words(text: str) -> str:
    toks = [t for t in tokenize(text) if t not in CARRIER_WORDS]
    return normalize(" ".join(toks))


def film_movie_phrase_variants(label: str) -> Set[str]:
    label = normalize(label)
    variants = {label}
    variants.add(label.replace(" film", " movie"))
    variants.add(label.replace(" films", " movies"))
    variants.add(label.replace(" movie", " film"))
    variants.add(label.replace(" movies", " films"))
    return {normalize(v) for v in variants if normalize(v)}


def build(dataset_dir: Path):
    artifacts_dir = get_runtime_artifacts_dir()
    id_entity_path = artifacts_dir / "id_entity.json"
    if not id_entity_path.exists():
        raise FileNotFoundError(f"Missing entity dictionary: {id_entity_path}")

    with id_entity_path.open("r", encoding="utf-8") as f:
        id_entity = json.load(f)

    # Prefer genre/tag-like entities so the synonym layer stays about descriptors,
    # not arbitrary named entities such as people, awards, or movie franchises.
    attr_index_path = artifacts_dir / "attr_index.pkl"
    descriptor_entity_ids = set()
    if attr_index_path.exists():
        with attr_index_path.open("rb") as f:
            attr_index = pickle.load(f)
        for attr, value_to_movies in attr_index.items():
            if attr == GENRE_RELATION or str(attr).startswith(ATAI_TAG_RELATION):
                descriptor_entity_ids.update(str(v) for v in value_to_movies.keys())

    if descriptor_entity_ids:
        canonical_labels = sorted({normalize(id_entity[k]) for k in descriptor_entity_ids if k in id_entity and normalize(id_entity[k])})
    else:
        canonical_labels = sorted({normalize(v) for v in id_entity.values() if normalize(v)})
    tokens = set()
    for label in canonical_labels:
        tokens.update(tokenize(label))

    wn = load_wordnet()
    word_to_synonyms = build_word_to_synonyms(tokens, wn)

    canonical_map: Dict[str, dict] = {}
    alias_to_canonical: Dict[str, str] = {}
    term_to_canonicals: Dict[str, Set[str]] = defaultdict(set)

    for canonical in canonical_labels:
        source_tokens = set(tokenize(canonical))
        term_set = set(source_tokens)
        surface_forms = set(film_movie_phrase_variants(canonical))
        surface_forms.update(PHRASE_ALIAS_SEEDS.get(canonical, set()))
        stripped_canonical = strip_carrier_words(canonical)
        if stripped_canonical and stripped_canonical != canonical:
            surface_forms.add(stripped_canonical)

        for token in list(source_tokens):
            term_set.update(word_to_synonyms.get(token, []))

        # Let seed phrase aliases contribute their tokens too.
        for alias in list(surface_forms):
            alias_norm = normalize(alias)
            if alias_norm:
                alias_to_canonical[alias_norm] = canonical
            alias_stripped = strip_carrier_words(alias)
            if alias_stripped:
                alias_to_canonical[alias_stripped] = canonical
            for tok in tokenize(alias):
                term_set.add(tok)
                term_set.update(word_to_synonyms.get(tok, []))

        canonical_map[canonical] = {
            "surface_forms": sorted(surface_forms | {canonical}),
            "source_tokens": sorted(source_tokens),
            "term_set": sorted(term_set),
        }

        for term in term_set:
            term_to_canonicals[term].add(canonical)
        for tok in source_tokens:
            term_to_canonicals[tok].add(canonical)

    combined = {
        "version": 1,
        "wordnet_available": wn is not None,
        "canonical_label_count": len(canonical_map),
        "token_count": len(word_to_synonyms),
        "word_to_synonyms": word_to_synonyms,
        "canonical_labels": canonical_map,
        "alias_to_canonical": {k: v for k, v in sorted(alias_to_canonical.items())},
        "term_to_canonicals": {k: sorted(v) for k, v in sorted(term_to_canonicals.items())},
    }

    entity_synonyms_path = artifacts_dir / "entity_synonyms.json"
    word_to_synonyms_path = artifacts_dir / "word_to_synonyms.json"

    with entity_synonyms_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False)
    with word_to_synonyms_path.open("w", encoding="utf-8") as f:
        json.dump(word_to_synonyms, f, ensure_ascii=False)

    print("Wrote synonym artifacts:")
    print(" -", entity_synonyms_path)
    print(" -", word_to_synonyms_path)
    print("Canonical labels:", len(canonical_map))
    print("Descriptor-filtered mode:", bool(descriptor_entity_ids))
    print("Token dictionary:", len(word_to_synonyms))
    print("WordNet available:", wn is not None)


if __name__ == "__main__":
    dataset = ensure_datapath()
    build(dataset)
