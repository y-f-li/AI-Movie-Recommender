from utils.message_blocks import debug_block, reply_block
from rdflib import Graph, URIRef
# Save .nt to cache for faster loading
from pathlib import Path
import pickle
import time
#/ Save .nt to cache for faster loading
import os

class KnowledgeBaseLoader:
    """Handles loading RDF/Turtle knowledge bases into an rdflib Graph."""

    def __init__(self, kb_path: str, format: str | None = None):
        self.kb_path = kb_path
        self.format = format or self._infer_format(self.kb_path)
        self.graph = Graph()

    def _infer_format(self, path: str) -> str:
        """
        Infer RDF serialization format from file extension.
        Currently supports only Turtle (.ttl) and N-Triples (.nt).
        """
        ext = os.path.splitext(path)[1].lower()

        if ext == ".ttl":
            return "turtle"
        elif ext == ".nt":
            return "nt"
        else:
            raise ValueError(
                f"Unsupported RDF format: {ext}. "
                "Currently supported formats are .ttl (Turtle) and .nt (N-Triples)."
            )

    def load(self, force_rebuild: bool = False, max_age_days: int | None = None):
        src = Path(self.kb_path)
        cache = src.with_suffix(".gpickle")   # any extension you like

        def cache_fresh() -> bool:
            if not cache.exists():
                return False
            if cache.stat().st_mtime < src.stat().st_mtime:
                return False
            if max_age_days is not None and (time.time() - cache.stat().st_mtime) > max_age_days*86400:
                return False
            return True

        if not force_rebuild and cache_fresh():
            with open(cache, "rb") as f:
                self.graph = pickle.load(f)     # ← fast load
            # DEBUG
            with debug_block():
                print(f"Loaded Knowledge base (cached) from '{cache.name}'.")
            # /DEBUG
            return

        # slow path exactly once
        self.graph.parse(location=str(src), format=(self.format or "nt11"))
        # DEBUG
        with debug_block():
            print(f"Loaded Knowledge base from '{src.name}'.")
        # /DEBUG

        # write cache for next time
        try:
            with open(cache, "wb") as f:
                pickle.dump(self.graph, f, protocol=pickle.HIGHEST_PROTOCOL)
            # DEBUG
            with debug_block():
                print(f"Wrote cache to '{cache.name}'.")
            # /DEBUG
        except Exception as e:
            # DEBUG
            with debug_block():
                print(f"Could not write cache: {e}")
            # /DEBUG

    def build_movie_attr_dict(self, rating_matrix, out_path):
        # Check if the graph object has been initialized
        if len(self.graph) == 0:
            with debug_block():
                print("Graph is empty")
            return

        movie_attrs: dict[str, set[str]] = {}
        # Extract the the list of movie id's from the matrix
        movie_ids: list[str] = rating_matrix.index.tolist()

        # In the movie attributes dictionary
        # For each movie collect all of the attributes And 
        # the form a new dictionary with attributes as keys and 
        # the actual value of the attributes as value of the attribute dictionary
        for qid in movie_ids:
            attrs_for_movie: dict[str, set[str]] = {}
            s = URIRef(qid)

            for p, o in self.graph.predicate_objects(s):
                pred = str(p)
                val = str(o)

                if pred not in attrs_for_movie:
                    attrs_for_movie[pred] = set()
                attrs_for_movie[pred].add(val)

            movie_attrs[qid] = attrs_for_movie

        # Save in pickle format
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as f:
            pickle.dump(movie_attrs, f, protocol=pickle.HIGHEST_PROTOCOL)
        with debug_block():
            print(f"Saved movie_attrs for {len(movie_attrs)} movies to {out_path}")

        return
    
    def build_attr_index(self, movie_attrs: dict[str, dict[str, set[str]]]):
        """
        Build:
        attr_index[attr][value_qid] = set of movie_qids
        e.g. attr_index["cast_member"]["Q38142"] = {"Q128504", "Q25188", ...}
        """
        attr_index: dict[str, dict[str, set[str]]] = {}

        for movie_qid, attrs in movie_attrs.items():
            for attr, values in attrs.items():
                for v in values:
                    attr_index.setdefault(attr, {}).setdefault(v, set()).add(movie_qid)

        return attr_index

    
class KnowledgeBaseLoaderDebugger:
    def __init__(self, dictionary):
        self.dictionary = dictionary
    def debug(self):
        with debug_block():
            print("Inception:\n")
            for key, value in self.dictionary['http://www.wikidata.org/entity/Q25188'].items():
                print(f"{key}: {value}")
                    