class PrefixExtractor:
    """
    Return a static prefix string for a given intent.
    Usage: PrefixExtractor().prefix_for(intent)
    """

    _MAP = {
        "director_of":        "I found the director to be ",
        "writer_of":          "I found the writer to be ",
        "release_date":       "the Release date you are looking for is ",
        "release_year":       "the Release year you requested is ",
        "cast_of":            "the Cast you are looking for is ",
        "movies_by_director": "the movies directed by the requested author are ",
        "movies_by_writer":   "Movies written by the requested writer are ",
        "movies_by_actor":    "Films starring the requested actor are ",
        "genre_of":           "the Genre is ",
        "language_of":        "the Language is ",
        "country_of":         "the Country of origin is ",
        "movies_by_tag":      "here is a collection of movies matching your tag:\n",
        "raw":                "",
    }

    def prefix_for(self, intent: str) -> str:
        return self._MAP.get(intent, self._MAP["raw"])
