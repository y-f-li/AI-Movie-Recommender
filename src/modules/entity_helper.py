from rdflib import Graph, RDFS, URIRef, Literal
from collections import Counter
from modules.title_resolver import TitleResolver
from utils.message_blocks import debug_block
import re

ATAI_TAG = URIRef("http://ddis.ch/atai/tag")

def get_en_label(g: Graph, iri: URIRef) -> str | None:
    """
    Return the English rdfs:label for a given IRI, if available.
    """
    for lbl in g.objects(iri, RDFS.label):
        if isinstance(lbl, Literal) and lbl.language == "en":
            return str(lbl)
    return None

def get_all_attributes_for_title(
    graph: Graph,
    title_resolver: TitleResolver,
    text: str,
) -> list[dict]:
    """
    Given some text containing a movie title (e.g. "I liked Inception"),
    resolve the title to a movie IRI and return all predicate–object
    pairs for that movie.
    """
    attributes: list[dict] = []

    # 1) text -> list of titles
    titles = title_resolver.extract_titles(text)

    with debug_block():
        print(f"[get_all_attributes_for_title] text: {text!r}")
        print(f"[get_all_attributes_for_title] matched titles: {titles}")

    if not titles:
        # no recognizable movie title in the text
        return attributes

    # 2) titles -> list of full IRIs (PREFIX_WD + QID)
    qids = title_resolver.qid_from_title(titles)

    if not qids:
        return attributes

    # 3) use the first movie IRI
    movie_iri_str = qids[0]
    movie_iri = URIRef(movie_iri_str)

    with debug_block():
        print(f"[get_all_attributes_for_title] using movie IRI: {movie_iri_str}")

    # 4) iterate over all predicate-object pairs for that subject
    for p, o in graph.predicate_objects(movie_iri):
        p_label = get_en_label(graph, p) if isinstance(p, URIRef) else None
        o_label = get_en_label(graph, o) if isinstance(o, URIRef) else None

        attributes.append(
            {
                "predicate": str(p),
                "predicate_label": p_label,
                "value": str(o),
                "value_label": o_label,
            }
        )

    return attributes

def get_movie_tags(graph, movie_iri: str) -> set[str]:
    """
    Return the atai tags for the given movie.
    """
    s = URIRef(movie_iri)
    tags = {str(tag) for tag in graph.objects(s, ATAI_TAG)}
    return tags

def build_tag_profile(graph, liked_movie_ids: list[str]) -> Counter:
    """
    Count how often each tag appears across liked movies.
    """
    tag_counter = Counter()

    for mid in liked_movie_ids:
        tags = get_movie_tags(graph, mid)
        tag_counter.update(tags)

    return tag_counter


def extract_wd_id(iri: str) -> str | None:
    """
    From something like
    '.../P1027'
    or '.../Q25188'
    return 'P1027' or 'Q25188'.
    If it doesn't end with P/Q + digits, return None.
    """
    id_pattern = re.compile(r"/([PQ]\d+)$")
    m = id_pattern.search(iri)
    return m.group(1) if m else None

def iri_to_label(query_executor, iri):
    p_prefix = "http://www.wikidata.org/prop/direct/"
    q_prefix = "http://www.wikidata.org/entity/"
    tail_id = extract_wd_id(iri)
    if not tail_id:
        pass
    elif tail_id[0] == 'P':
        iri = p_prefix + tail_id
    elif tail_id[0] == 'Q':
        iri = q_prefix + tail_id
    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?label WHERE {{
    <{iri}> rdfs:label ?label .
    }}
    LIMIT 1
    """
    result, _ = query_executor.execute_query(query)
    return result[0]

class EntityLabelIndex:
    def __init__(self, graph):
        self.graph = graph
        self.label_to_qid: dict[str, str] = {}
        self.qid_to_label: dict[str, str] = {}

    def add_entity(self, iri: URIRef | str):
        if isinstance(iri, URIRef):
            label = self.graph.value(iri, RDFS.label)
            if label is None:
                return
            label_str = str(label)
            qid = iri.split("/")[-1]
            key = label_str.lower()
            self.label_to_qid[key] = qid
            self.qid_to_label[qid] = label_str
        else:
            label = iri
            label_str = str(label)
            qid = iri
            key = label_str.lower()
            self.label_to_qid[key] = qid
            self.qid_to_label[qid] = label_str