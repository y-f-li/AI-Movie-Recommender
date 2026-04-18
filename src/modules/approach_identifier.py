import re, unicodedata
from enum import Enum
from utils.message_blocks import debug_block

class Approach(Enum):
    EMBEDDING = "embedding"
    FACTUAL = "factual"
    RECOMMENDATION = "recommendation"
    MULTIMEDIA = "multimedia"

class ApproachIdentifier:
    def _norm(self, s: str) -> str:
        """
        Normalize the text:
        - Convert fancy Unicode symbols into standard ones (NFKC)
        - Collapse multiple spaces into one
        """
        s = unicodedata.normalize("NFKC", s or "")
        return " ".join(s.split())

    PREFIX = re.compile(
        r'^\s*(?:could\s+you|would\s+you|please|kindly|hey|hi|hello|ok[,!]?)?\s*'
        r'(?:.*?\b(?:answer|respond)\b)?\s*'
        r'(?:.*?\bwith\b\s+(?:an?\s+)?(?P<mode>embedding|factual)\b(?:\s+approach)?\s*)?'
        r'[:\-–—]?\s*',
        re.IGNORECASE
    )

    def identify_approach(self, message: str):
        """
        Identify the high-level route for a message.

        Returns:
            (approach, explicitly_requested)
        where explicitly_requested is True only when the user asked for a
        specific approach mode such as "with factual approach".
        """
        msg = self._norm(message)
        prefix_match = self.PREFIX.match(msg)
        if prefix_match and prefix_match.group("mode"):
            mode = prefix_match.group("mode").lower()
            approach = {
                "embedding": Approach.EMBEDDING,
                "factual": Approach.FACTUAL,
            }.get(mode, Approach.RECOMMENDATION)
            with debug_block():
                print(f"Explicit approach requested: {approach.value}")
            return approach, True

        if self.is_a_multimedia_request(msg):
            with debug_block():
                print("Detected multimedia request.")
            return Approach.MULTIMEDIA, False

        if self.is_a_recommendation_request(msg):
            with debug_block():
                print("Detected recommendation request.")
            return Approach.RECOMMENDATION, False

        if self.is_a_factual_request(msg):
            with debug_block():
                print("Detected factual request.")
            return Approach.FACTUAL, False

        with debug_block():
            print("No strong approach signal detected; defaulting to recommendation.")
        return Approach.RECOMMENDATION, False

    def is_a_recommendation_request(self, message: str) -> bool:
        low = message.lower()

        keywords = [
            "recommend", "suggest", "what should i watch", "any good movies",
            "looking for movies", "i want to watch", "can you suggest",
            "can you recommend", "what to watch", "movie ideas",
            "got any movies", "need something to watch", "any suggestions",
            "anything good to watch", "watch next", "similar movies", "similar", "i like", "i love",
            "show me", "I want", "I watch"
        ]

        return any(kw in low for kw in keywords)

    def is_a_factual_request(self, message: str) -> bool:
        low = message.lower().strip()

        question_starters = (
            "who ", "who's ", "who is ",
            "what ", "what's ", "what is ",
            "when ", "when was ",
            "where ", "where was ", "where does ",
            "which ", "which movies ",
            "how much ", "how many ",
            "list ", "name the ",
        )

        factual_cues = [
            "director", "writer", "screenwriter", "cast", "genre", "language", "country",
            "release date", "release year", "box office", "composer", "cinematographer",
            "editor", "filming location", "take place", "based on", "mpaa", "rating",
            "main subject", "distributor", "voice actor", "character", "series",
            "awards", "nominated", "produced", "production company",
        ]

        if low.endswith("?") and any(cue in low for cue in factual_cues):
            return True

        if any(low.startswith(starter) for starter in question_starters) and any(cue in low for cue in factual_cues):
            return True

        return False

    def is_a_multimedia_request(self, message: str) -> bool:
        """
        Multimedia is intentionally frozen in this iteration.
        We keep the method only as a compatibility stub so the rest of the
        codebase can be simplified later without breaking imports.
        """
        return False

        keywords = [
            "show", "image", "picture", "photo", "trailer", "video", "clip", "look like", "looks like", "poster", "backdrop"
        ]

        return any(kw in low for kw in keywords)