import re
from typing import Dict, List, Tuple, Pattern

TITLE_CORE = r'[“"\'‘’]?(.+?)[”"\'’]?'
TITLE = r'[“"\'‘’]?(.+?)[”"\'’]?(?=\s*(?:\?|$))'
OPTIONAL_PREFIX = r"(?:.*?:\s*)?"
MOVIE_PREFIX = r"(?:(?:the\s+)?(?:movie|film)\s+)?"

class PatternLibrary: 

    def __init__(self):
        self.patterns = [
            # Director related patterns: 
    		(re.compile(rf"who\s+is\s+the\s+director\s+of\s+{TITLE}$", re.I), "director_of"),            
            (re.compile(rf"who\s+directed\s+{TITLE}$", re.I), "director_of"),
            (re.compile(rf"^\s*who(?:'s|\s+is)\s+(?:the\s+)?director\s+(?:of|for)\s+{TITLE}\s*\??\s*$", re.I), "director_of"),
            (re.compile(rf"^\s*who\s+directed\s+{TITLE}\s*\??\s*$", re.I), "director_of"),
            (re.compile(rf"^\s*who\s+is\s+{TITLE}\s+directed\s+by\s*\??\s*$", re.I), "director_of"),
            (re.compile(rf"^\s*{TITLE}\s+was\s+directed\s+by\s+who\s*\??\s*$", re.I), "director_of"),
            (re.compile(rf"^\s*name\s+of\s+(?:the\s+)?director\s+(?:of|for)\s+{TITLE}\s*\??\s*$", re.I), "director_of"),
            (re.compile(rf"^\s*director\s+(?:of|for)\s+{TITLE}\s*\??\s*$", re.I), "director_of"),
            (re.compile(rf"^\s*who\s+was\s+(?:the\s+)?director\s+(?:of|for)\s+{TITLE}\s*\??\s*$", re.I), "director_of"),
            (re.compile(rf"^\s*who\s+is\s+credited\s+as\s+(?:the\s+)?director\s+(?:of|for)\s+{TITLE}\s*\??\s*$", re.I), "director_of"),
            (re.compile(rf"^\s*who\s+helmed\s+{TITLE}\s*\??\s*$", re.I), "director_of"),
            (re.compile(rf"^\s*who\s+is\s+(?:the\s+)?filmmaker\s+behind\s+{TITLE}\s*\??\s*$", re.I), "director_of"),
            (re.compile(rf"^\s*who\s+is\s+(?:the\s+)?film\s+director\s+(?:of|for)\s+{TITLE}\s*\??\s*$", re.I), "director_of"),
            (re.compile(rf"^\s*who\s+was\s+at\s+the\s+helm\s+of\s+{TITLE}\s*\??\s*$", re.I), "director_of"),
            (re.compile(rf"who\s+did\s+direct\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "director_of"),
            (re.compile(rf"^\s*who\s+(?:is|was)\s+{TITLE_CORE}\s+directed\s+by\s*\??\s*$", re.I), "director_of"),
            # Writer related patterns: 
            (re.compile(rf"^{OPTIONAL_PREFIX}\s*who\s+is\s+(?:the\s+)?screenwriter\s+of\s+{TITLE}\s*\??\s*$", re.I), "writer_of"),
            (re.compile(rf"^{OPTIONAL_PREFIX}\s*who\s+is\s+(?:the\s+)?writer\s+of\s+{TITLE}\s*\??\s*$", re.I), "writer_of"),
            (re.compile(rf"^{OPTIONAL_PREFIX}\s*who(?:'s|\s+is)\s+(?:the\s+)?screenwriter\s+of\s+{TITLE}\s*\??\s*$", re.I), "writer_of"),
            (re.compile(rf"^{OPTIONAL_PREFIX}\s*who(?:'s|\s+is)\s+(?:the\s+)?writer\s+of\s+{TITLE}\s*\??\s*$", re.I), "writer_of"),
            (re.compile(rf"^{OPTIONAL_PREFIX}\s*who\s+wrote\s+(?:the\s+)?screenplay\s+for\s+{TITLE}\s*\??\s*$", re.I), "writer_of"),
            (re.compile(rf"^{OPTIONAL_PREFIX}\s*who\s+wrote\s+(?:the\s+)?script\s+for\s+{TITLE}\s*\??\s*$", re.I), "writer_of"),
            (re.compile(rf"^{OPTIONAL_PREFIX}\s*who\s+is\s+{TITLE}\s+written\s+by\s*\??\s*$", re.I), "writer_of"),
            (re.compile(rf"^{OPTIONAL_PREFIX}\s*{TITLE}\s+was\s+written\s+by\s+who\s*\??\s*$", re.I), "writer_of"),
            (re.compile(rf"^{OPTIONAL_PREFIX}\s*who\s+authored\s+(?:the\s+)?screenplay\s+for\s+{TITLE}\s*\??\s*$", re.I), "writer_of"),
            (re.compile(rf"^{OPTIONAL_PREFIX}\s*who\s+penned\s+(?:the\s+)?screenplay\s+for\s+{TITLE}\s*\??\s*$", re.I), "writer_of"),
            (re.compile(rf"^{OPTIONAL_PREFIX}\s*who\s+is\s+credited\s+with\s+writing\s+(?:the\s+)?(?:screenplay|script)\s+for\s+{TITLE}\s*\??\s*$", re.I), "writer_of"),
            (re.compile(rf"^{OPTIONAL_PREFIX}\s*who\s+did\s+(?:the\s+)?writing\s+for\s+{TITLE}\s*\??\s*$", re.I), "writer_of"),
            (re.compile(rf"who\s+did\s+write\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "writer_of"),
            (re.compile(rf"who\s+did\s+write\s+(?:the\s+)?screenplay\s+(?:for|of)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "writer_of"),

            # Release date related patterns: 
            (re.compile(rf"when\s+was\s+{TITLE_CORE}\s+released\??\s*$", re.I), "release_year"),
            (re.compile(rf"what\s+year\s+did\s+{TITLE_CORE}\s+come\s+out\??\s*$", re.I), "release_year"),
            (re.compile(rf"when\s+did\s+the\s+movie\s+{TITLE_CORE}\s+come\s+out\??\s*$", re.I), "release_year"),
            (re.compile(rf"^\s*what\s+is\s+(?:the\s+)?release\s+date\s+of\s+{TITLE_CORE}\s*\??\s*$", re.I), "release_year"),
            (re.compile(rf"^\s*what'?s\s+(?:the\s+)?release\s+date\s+(?:of|for)\s+{TITLE_CORE}\s*\??\s*$", re.I), "release_year"),
            (re.compile(rf"^\s*release\s+date\s+(?:of|for)\s+{TITLE_CORE}\s*\??\s*$", re.I), "release_year"),
            (re.compile(rf"^\s*when\s+was\s+(?:the\s+)?(?:movie|film)\s+{TITLE_CORE}\s+released\s*\??\s*$", re.I), "release_year"),
            (re.compile(rf"^\s*what\s+year\s+was\s+{TITLE_CORE}\s+released\s*\??\s*$", re.I), "release_year"),
            (re.compile(rf"^\s*when\s+did\s+{TITLE_CORE}\s+(?:release|come\s+out|premiere|debut)\s*\??\s*$", re.I), "release_year"),
            (re.compile(rf"^\s*when\s+did\s+(?:the\s+)?(?:movie|film)\s+{TITLE_CORE}\s+(?:premiere|debut)\s*\??\s*$", re.I), "release_year"),
            (re.compile(rf"^\s*when\s+did\s+{TITLE_CORE}\s+open\s+in\s+the(?:aters|atres)\s*\??\s*$", re.I), "release_year"),
            (re.compile(rf"^\s*when\s+did\s+{TITLE_CORE}\s+hit\s+the\s+the(?:aters|atres)\s*\??\s*$", re.I), "release_year"),
            (re.compile(rf"^\s*when\s+was\s+{TITLE_CORE}\s+first\s+(?:shown|released)\s+(?:in\s+cine(?:ma|mas)\s*)?\??\s*$", re.I), "release_year"),


            # Cast related patterns: 
            (re.compile(rf'who\s+starred\s+in\s+{TITLE}$', re.I), "cast_of"),
            (re.compile(rf'list\s+the\s+cast\s+of\s+{TITLE}$', re.I), "cast_of"),
            (re.compile(rf'who\s+is\s+in\s+the\s+cast\s+of\s+{TITLE}$', re.I), "cast_of"),
            (re.compile(rf'who\s+is\s+in\s+{TITLE}$', re.I), "cast_of"),
            (re.compile(rf'which\s+actors\s+are\s+in\s+{TITLE}$', re.I), "cast_of"),
            (re.compile(rf'who\s+stars?\s+in\s+{TITLE}$', re.I), "cast_of"),
            (re.compile(rf'(?:the\s+)?main\s+cast\s+of\s+{TITLE}$', re.I), "cast_of"),
            (re.compile(rf'(?:the\s+)?full\s+cast\s+of\s+{TITLE}$', re.I), "cast_of"),
            (re.compile(rf'(?:the\s+)?voice\s+cast\s+(?:of|in)\s+{TITLE}$', re.I), "cast_of"),
            (re.compile(rf'{TITLE_CORE}\s+cast\??\s*$', re.I), "cast_of"),
            (re.compile(rf'{TITLE_CORE}\s+actors\??\s*$', re.I), "cast_of"),
            (re.compile(rf'what\s+is\s+the\s+cast\s+of\s+{TITLE}$', re.I), "cast_of"),
            (re.compile(rf'who\s+casted\s+in\s+{TITLE}$', re.I), "cast_of"),

            # Director/Writer Legacy pattersn: 
            (re.compile(r"which\s+movies\s+did\s+(.+?)\s+direct\??", re.I), "movies_by_director"),
            (re.compile(r"which\s+movies\s+did\s+(.+?)\s+write\??", re.I), "movies_by_writer"),
            (re.compile(r"which\s+movies\s+did\s+(.+?)\s+star\s+in\??", re.I), "movies_by_actor"),
            (re.compile(r"which\s+movies\s+did\s+(.+?)\s+produce\??", re.I), "movies_by_producer"),
            (re.compile(r"which\s+movies\s+did\s+(.+?)\s+compose\??", re.I), "movies_by_composer"),
            (re.compile(r"which\s+movies\s+did\s+(.+?)\s+edit\??", re.I), "movies_by_editor"),
            (re.compile(r"which\s+movies\s+did\s+(.+?)\s+shoot\??", re.I), "movies_by_cinematographer"),


            # Genre pattersn: 
            (re.compile(rf"what\s+is\s+the\s+genre\s+of\s+{TITLE}$", re.I), "genre_of"),
            (re.compile(rf"what(?:'s|\s+is)\s+the\s+genre\s+of\s+{TITLE}$", re.I), "genre_of"),
            (re.compile(rf"what\s+genre\s+is\s+{TITLE}$", re.I), "genre_of"),
            (re.compile(rf"which\s+genre\s+is\s+{TITLE}$", re.I), "genre_of"),
            (re.compile(rf"genre\s+(?:of|for)\s+{TITLE}$", re.I), "genre_of"),
            (re.compile(rf"the\s+genre\s+of\s+{TITLE}$", re.I), "genre_of"),
            (re.compile(rf"{TITLE_CORE}\s+genre\??\s*$", re.I), "genre_of"),
            (re.compile(rf"what\s+genre\s+does\s+{TITLE_CORE}\s+belong\s+to\??\s*$", re.I), "genre_of"),
            (re.compile(rf"what\s+genre\s+would\s+you\s+classify\s+{TITLE_CORE}\s+as\??\s*$", re.I), "genre_of"),
            (re.compile(rf"{TITLE_CORE}\s+falls?\s+under\s+which\s+genre\??\s*$", re.I), "genre_of"),
            (re.compile(rf"what\s+(?:type|kind)\s+of\s+movie\s+is\s+{TITLE}$", re.I), "genre_of"),
            (re.compile(rf"{TITLE_CORE}\s+is\s+what\s+(?:type|kind)\s+of\s+movie\??\s*$", re.I), "genre_of"), 


            # Language pattersn: 
            (re.compile(rf"what\s+language\s+is\s+{TITLE_CORE}\s+in\??\s*$", re.I), "language_of"),
            (re.compile(rf"^\s*which\s+language\s+is\s+{TITLE_CORE}\s+in\??\s*$", re.I), "language_of"),
            (re.compile(rf"^\s*what'?s\s+(?:the\s+)?language\s+(?:of|for)\s+{TITLE_CORE}\s*\??\s*$", re.I), "language_of"),
            (re.compile(rf"^\s*language\s+(?:of|for)\s+{TITLE_CORE}\s*\??\s*$", re.I), "language_of"),
            (re.compile(rf"^\s*in\s+what\s+language\s+is\s+(?:the\s+)?(?:movie|film)?\s*{TITLE_CORE}\s*\??\s*$", re.I), "language_of"),
            (re.compile(rf"^\s*what\s+is\s+(?:the\s+)?original\s+language\s+(?:of|for)\s+{TITLE_CORE}\s*\??\s*$", re.I), "language_of"),
            (re.compile(rf"^\s*which\s+language\s+was\s+{TITLE_CORE}\s+(?:made|filmed|produced)\s+in\s*\??\s*$", re.I), "language_of"),
            (re.compile(rf"^\s*what\s+language\s+does\s+{TITLE_CORE}\s+use\??\s*$", re.I), "language_of"),
            (re.compile(rf"^\s*what\s+language\s+is\s+(?:the\s+)?(?:movie|film)\s+{TITLE_CORE}\s+in\s*\??\s*$", re.I), "language_of"),
            (re.compile(rf"^\s*what\s+language\s+is\s+{TITLE_CORE}\s+(?:spoken|performed)\s+in\s*\??\s*$", re.I), "language_of"),
            (re.compile(rf"^\s*what\s+language\s+do\s+they\s+speak\s+in\s+{TITLE_CORE}\s*\??\s*$", re.I), "language_of"),


            # Country Patterns: 
            (re.compile(rf"which\s+country\s+is\s+{TITLE_CORE}\s+from\??\s*$", re.I), "country_of"),
            (re.compile(rf"what\s+country\s+is\s+{TITLE_CORE}\s+from\??\s*$", re.I), "country_of"),
            (re.compile(rf"{TITLE_CORE}\s+is\s+from\s+which\s+country\??\s*$", re.I), "country_of"),
            (re.compile(rf"which\s+country\s+does\s+{TITLE_CORE}\s+come\s+from\??\s*$", re.I), "country_of"),
            (re.compile(rf"from\s+which\s+country\s+is\s+{TITLE_CORE}\??\s*$", re.I), "country_of"),
            (re.compile(rf"from\s+what\s+country\s+is\s+(?:the\s+)?(?:movie|film)\s+{TITLE_CORE}\??\s*$", re.I), "country_of"),
            (re.compile(rf"what\s+is\s+the\s+country\s+of\s+origin\s+of\s+{TITLE}$", re.I), "country_of"),
            (re.compile(rf"the\s+country\s+of\s+origin\s+of\s+{TITLE}$", re.I), "country_of"),
            (re.compile(rf"{TITLE_CORE}\s+country\s+of\s+origin\??\s*$", re.I), "country_of"),
            (re.compile(rf"origin\s+country\s+of\s+{TITLE}$", re.I), "country_of"),
            (re.compile(rf"which\s+country\s+produced\s+{TITLE}$", re.I), "country_of"),
            (re.compile(rf"what\s+country\s+produced\s+{TITLE}$", re.I), "country_of"),
            (re.compile(rf"{TITLE_CORE}\s+was\s+produced\s+in\s+which\s+country\??\s*$", re.I), "country_of"),
            (re.compile(rf"which\s+country\s+made\s+{TITLE}$", re.I), "country_of"),
            (re.compile(rf"what\s+country\s+made\s+{TITLE}$", re.I), "country_of"),
            (re.compile(rf"where\s+is\s+{TITLE_CORE}\s+from\??\s*$", re.I), "country_of"),
            (re.compile(rf"{TITLE_CORE}\s+is\s+a\s+film\s+from\s+which\s+country\??\s*$", re.I), "country_of"),
            (re.compile(rf"{TITLE_CORE}\s+belongs?\s+to\s+which\s+country(?:'s)?\s+cinema\??\s*$", re.I), "country_of"),


            (re.compile(r"movies\s+about\s+(.+)", re.I), "movies_by_tag"),
            
            #  1. PRODUCER (P162) 
            (re.compile(rf"who\s+produced\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "producer_of"),
            (re.compile(rf"who\s+is\s+the\s+producer\s+(?:of|for)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "producer_of"),
            (re.compile(rf"who\s+was\s+the\s+producer\s+(?:of|for)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "producer_of"),
            (re.compile(rf"who\s+made\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "producer_of"), # ambiguous but often means producer
            (re.compile(rf"name\s+the\s+producer\s+(?:of|for)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "producer_of"),
            (re.compile(rf"{MOVIE_PREFIX}{TITLE_CORE}\s+producer\??\s*$", re.I), "producer_of"),
            (re.compile(rf"who\s+served\s+as\s+producer\s+on\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "producer_of"),

            #  2. PRODUCTION COMPANY (P272) 
            (re.compile(rf"which\s+company\s+produced\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "production_company"),
            (re.compile(rf"what\s+studio\s+made\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "production_company"),
            (re.compile(rf"what\s+company\s+is\s+behind\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "production_company"),
            (re.compile(rf"production\s+company\s+(?:of|for)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "production_company"),
            (re.compile(rf"which\s+studio\s+released\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "production_company"), # Often synonymous in user intent
            (re.compile(rf"{MOVIE_PREFIX}{TITLE_CORE}\s+production\s+company\??\s*$", re.I), "production_company"),

            #  3. AWARDS (P166) 

            (re.compile(rf"what\s+awards?\s+(?:did|has)\s+{MOVIE_PREFIX}{TITLE_CORE}\s+(?:won|received|got|earned|garnered)\??\s*$", re.I), "award_received"),
            (re.compile(rf"which\s+awards?\s+(?:did|has)\s+{MOVIE_PREFIX}{TITLE_CORE}\s+(?:won|received|got|earned)\??\s*$", re.I), "award_received"),
            (re.compile(rf"did\s+{MOVIE_PREFIX}{TITLE_CORE}\s+win\s+any\s+awards?\??\s*$", re.I), "award_received"),
            (re.compile(rf"list\s+(?:the\s+)?awards?\s+(?:for|of)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "award_received"),
            (re.compile(rf"awards?\s+won\s+by\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "award_received"),
            (re.compile(rf"what\s+accolades?\s+did\s+{MOVIE_PREFIX}{TITLE_CORE}\s+get\??\s*$", re.I), "award_received"),
            (re.compile(rf"{MOVIE_PREFIX}{TITLE_CORE}\s+awards?\??\s*$", re.I), "award_received"),
            (re.compile(rf"what\s+awards?\s+did\s+{MOVIE_PREFIX}{TITLE_CORE}\s+win\??\s*$", re.I), "award_received"),
            (re.compile(rf"which\s+awards?\s+did\s+{MOVIE_PREFIX}{TITLE_CORE}\s+win\??\s*$", re.I), "award_received"),
            (re.compile(rf"did\s+{MOVIE_PREFIX}{TITLE_CORE}\s+win\s+the\s+(.+?)\??\s*$", re.I), "award_received"), 

            #  4. NOMINATIONS (P1411) 
            (re.compile(rf"what\s+awards?\s+was\s+{MOVIE_PREFIX}{TITLE_CORE}\s+nominated\s+for\??\s*$", re.I), "nominated_for"),
            (re.compile(rf"which\s+nominations?\s+did\s+{MOVIE_PREFIX}{TITLE_CORE}\s+(?:get|receive|have)\??\s*$", re.I), "nominated_for"),
            (re.compile(rf"was\s+{MOVIE_PREFIX}{TITLE_CORE}\s+nominated\s+for\s+any\s+awards?\??\s*$", re.I), "nominated_for"),
            (re.compile(rf"list\s+(?:the\s+)?nominations?\s+(?:for|of)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "nominated_for"),
            (re.compile(rf"what\s+is\s+{MOVIE_PREFIX}{TITLE_CORE}\s+nominated\s+for\??\s*$", re.I), "nominated_for"),
            (re.compile(rf"what\s+awards?\s+was\s+{MOVIE_PREFIX}{TITLE_CORE}\s+nominated\s+for\??\s*$", re.I), "nominated_for"),
            (re.compile(rf"which\s+awards?\s+was\s+{MOVIE_PREFIX}{TITLE_CORE}\s+nominated\s+for\??\s*$", re.I), "nominated_for"),
            (re.compile(rf"was\s+{MOVIE_PREFIX}{TITLE_CORE}\s+nominated\s+for\s+any\s+awards?\??\s*$", re.I), "nominated_for"),

            #  5. BOX OFFICE / REVENUE (P2142) 
            (re.compile(rf"how\s+much\s+(?:money\s+)?did\s+{MOVIE_PREFIX}{TITLE_CORE}\s+(?:make|earn|gross|bring\s+in)\??\s*$", re.I), "box_office"),
            (re.compile(rf"what\s+was\s+the\s+box\s+office\s+(?:for|of)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "box_office"),
            (re.compile(rf"what\s+is\s+the\s+(?:gross|total)\s+revenue\s+(?:of|for)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "box_office"),
            (re.compile(rf"box\s+office\s+(?:sales|results|earnings)\s+(?:for|of)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "box_office"),
            (re.compile(rf"{MOVIE_PREFIX}{TITLE_CORE}\s+box\s+office\??\s*$", re.I), "box_office"),
            (re.compile(rf"how\s+successful\s+was\s+{MOVIE_PREFIX}{TITLE_CORE}\s+financially\??\s*$", re.I), "box_office"),

            #  6. COMPOSER / MUSIC (P86) 
            (re.compile(rf"who\s+(?:composed|wrote)\s+the\s+music\s+(?:for|of)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "composer"),
            (re.compile(rf"who\s+is\s+the\s+composer\s+(?:for|of)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "composer"),
            (re.compile(rf"who\s+did\s+the\s+(?:soundtrack|score)\s+(?:for|of)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "composer"),
            (re.compile(rf"who\s+scored\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "composer"),
            (re.compile(rf"music\s+by\s+whom\s+in\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "composer"),
            (re.compile(rf"{MOVIE_PREFIX}{TITLE_CORE}\s+composer\??\s*$", re.I), "composer"),
            (re.compile(rf"who\s+did\s+compose\s+(?:the\s+)?music\s+(?:for|of)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "composer"),
            (re.compile(rf"who\s+did\s+compose\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "composer"),

            #  7. CINEMATOGRAPHER (P344) 
            (re.compile(rf"who\s+shot\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cinematographer"),
            (re.compile(rf"who\s+was\s+the\s+cinematographer\s+(?:for|of)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cinematographer"),
            (re.compile(rf"who\s+is\s+the\s+director\s+of\s+photography\s+(?:for|of)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cinematographer"),
            (re.compile(rf"who\s+filmed\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cinematographer"),
            (re.compile(rf"who\s+was\s+behind\s+the\s+camera\s+for\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cinematographer"),
            (re.compile(rf"{MOVIE_PREFIX}{TITLE_CORE}\s+cinematography\??\s*$", re.I), "cinematographer"),

            #  8. EDITOR (P1040) 
            (re.compile(rf"who\s+edited\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "editor"),
            (re.compile(rf"who\s+was\s+the\s+editor\s+(?:of|for)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "editor"),
            (re.compile(rf"who\s+did\s+the\s+editing\s+for\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "editor"),
            (re.compile(rf"who\s+cut\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "editor"),
            (re.compile(rf"{MOVIE_PREFIX}{TITLE_CORE}\s+editor\??\s*$", re.I), "editor"),

            #  9. FILMING LOCATION (P915) 
            (re.compile(rf"where\s+was\s+{MOVIE_PREFIX}{TITLE_CORE}\s+filmed\??\s*$", re.I), "filming_location"),
            (re.compile(rf"where\s+did\s+they\s+film\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "filming_location"),
            (re.compile(rf"what\s+are\s+the\s+filming\s+locations\s+(?:of|for)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "filming_location"),
            (re.compile(rf"location\s+of\s+filming\s+(?:for|of)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "filming_location"),
            (re.compile(rf"where\s+was\s+{MOVIE_PREFIX}{TITLE_CORE}\s+shot\??\s*$", re.I), "filming_location"),

            #  10. NARRATIVE LOCATION / SETTING (P840) 
            (re.compile(rf"where\s+does\s+{MOVIE_PREFIX}{TITLE_CORE}\s+take\s+place\??\s*$", re.I), "narrative_location"),
            (re.compile(rf"what\s+is\s+the\s+setting\s+(?:of|for)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "narrative_location"),
            (re.compile(rf"where\s+is\s+{MOVIE_PREFIX}{TITLE_CORE}\s+set\??\s*$", re.I), "narrative_location"),
            (re.compile(rf"in\s+which\s+city\s+is\s+{MOVIE_PREFIX}{TITLE_CORE}\s+set\??\s*$", re.I), "narrative_location"),
            (re.compile(rf"location\s+of\s+the\s+story\s+in\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "narrative_location"),

            #  11. BASED ON (P144) 
            (re.compile(rf"what\s+is\s+{MOVIE_PREFIX}{TITLE_CORE}\s+based\s+on\??\s*$", re.I), "based_on"),
            (re.compile(rf"is\s+{MOVIE_PREFIX}{TITLE_CORE}\s+based\s+on\s+a\s+(?:book|novel|true\s+story)\??\s*$", re.I), "based_on"),
            (re.compile(rf"what\s+book\s+is\s+{MOVIE_PREFIX}{TITLE_CORE}\s+based\s+on\??\s*$", re.I), "based_on"),
            (re.compile(rf"source\s+material\s+for\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "based_on"),
            (re.compile(rf"did\s+{MOVIE_PREFIX}{TITLE_CORE}\s+come\s+from\s+a\s+book\??\s*$", re.I), "based_on"),

            #  12. MPAA RATING (P1657) 
            (re.compile(rf"what\s+is\s+the\s+(?:mpaa\s+)?rating\s+(?:of|for)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "mpaa_rating"),
            (re.compile(rf"is\s+{MOVIE_PREFIX}{TITLE_CORE}\s+suitable\s+for\s+children\??\s*$", re.I), "mpaa_rating"),
            (re.compile(rf"what\s+is\s+{MOVIE_PREFIX}{TITLE_CORE}\s+rated\??\s*$", re.I), "mpaa_rating"),
            (re.compile(rf"age\s+rating\s+(?:for|of)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "mpaa_rating"),
            (re.compile(rf"can\s+kids\s+watch\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "mpaa_rating"),

            #  13. MAIN SUBJECT / ABOUT (P921) 
            (re.compile(rf"what\s+is\s+{MOVIE_PREFIX}{TITLE_CORE}\s+about\??\s*$", re.I), "main_subject"),
            (re.compile(rf"what\s+is\s+the\s+(?:main\s+)?subject\s+(?:of|for)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "main_subject"),
            (re.compile(rf"what\s+is\s+the\s+plot\s+(?:of|for)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "main_subject"),
            (re.compile(rf"describe\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "main_subject"),
            (re.compile(rf"what\s+happens\s+in\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "main_subject"),

            # 14. DISTRIBUTOR (P750) 
            (re.compile(rf"who\s+distributed\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "distributor"),
            (re.compile(rf"which\s+company\s+distributed\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "distributor"),
            (re.compile(rf"who\s+released\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "distributor"), # can overlap with production
            (re.compile(rf"distributor\s+(?:of|for)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "distributor"),
            
            # 14. CAST (P161) 
            (re.compile(rf"who\s+is\s+a\s+cast\s+member\s+(?:of|in)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cast_of"),
            (re.compile(rf"list\s+(?:the\s+)?cast\s+members\s+(?:of|for)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cast_of"),
            (re.compile(rf"which\s+cast\s+members\s+are\s+in\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cast_of"),
            (re.compile(rf"show\s+me\s+the\s+cast\s+members\s+(?:of|from)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cast_of"),
            (re.compile(rf"{MOVIE_PREFIX}{TITLE_CORE}\s+cast\s+members\??\s*$", re.I), "cast_of"),
            (re.compile(rf"who\s+is\s+in\s+the\s+cast\s+of\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cast_of"),
            (re.compile(rf"who\s+is\s+part\s+of\s+the\s+cast\s+of\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cast_of"),
            (re.compile(rf"what\s+is\s+the\s+cast\s+of\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cast_of"),
            (re.compile(rf"what\s+is\s+the\s+cast\s+list\s+(?:for|of)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cast_of"),
            (re.compile(rf"list\s+(?:the\s+)?cast\s+(?:of|for)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cast_of"),
            (re.compile(rf"give\s+me\s+the\s+cast\s+(?:of|for)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cast_of"),
            (re.compile(rf"who\s+appeared\s+in\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cast_of"),
            (re.compile(rf"who\s+appears\s+in\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cast_of"),
            (re.compile(rf"who\s+starred\s+in\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cast_of"),
            (re.compile(rf"who\s+stars\s+in\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cast_of"),
            (re.compile(rf"who\s+co-starred\s+in\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cast_of"),
            (re.compile(rf"who\s+acted\s+in\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cast_of"),
            (re.compile(rf"who\s+plays\s+in\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cast_of"),
            (re.compile(rf"who\s+played\s+in\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cast_of"),
            (re.compile(rf"who\s+has\s+a\s+role\s+in\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cast_of"),
            (re.compile(rf"who\s+had\s+a\s+role\s+in\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cast_of"),
            (re.compile(rf"who\s+are\s+the\s+actors\s+in\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cast_of"),
            (re.compile(rf"which\s+actors\s+are\s+in\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cast_of"),
            (re.compile(rf"list\s+the\s+actors\s+(?:in|of)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cast_of"),
            (re.compile(rf"actors\s+in\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cast_of"),
            (re.compile(rf"who\s+is\s+a\s+cast\s+member\s+(?:of|in)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cast_of"),
            (re.compile(rf"list\s+(?:the\s+)?cast\s+members\s+(?:of|for)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cast_of"),
            (re.compile(rf"which\s+cast\s+members\s+are\s+in\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cast_of"),
            (re.compile(rf"show\s+me\s+the\s+cast\s+members\s+(?:of|from)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cast_of"),
            (re.compile(rf"{MOVIE_PREFIX}{TITLE_CORE}\s+cast\s+members\??\s*$", re.I), "cast_of"),
            (re.compile(rf"{MOVIE_PREFIX}{TITLE_CORE}\s+cast\??\s*$", re.I), "cast_of"),
            (re.compile(rf"cast\s+of\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "cast_of"),

            # --- 15. VOICE ACTOR (P725) ---
            (re.compile(rf"who\s+voiced\s+(?:the\s+characters?\s+in\s+)?{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "voice_actor"),
            (re.compile(rf"who\s+is\s+the\s+voice\s+(?:of|for|in)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "voice_actor"),
            (re.compile(rf"who\s+does\s+the\s+voice\s+(?:acting\s+)?in\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "voice_actor"),

            # --- 16. CHARACTERS (P674) ---
            (re.compile(rf"what\s+characters\s+are\s+in\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "characters_in"),
            (re.compile(rf"who\s+are\s+the\s+characters\s+in\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "characters_in"),
            (re.compile(rf"list\s+(?:the\s+)?characters\s+(?:of|in)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "characters_in"),

            # --- 17. SERIES / FRANCHISE (P179) ---
            (re.compile(rf"what\s+series\s+is\s+{MOVIE_PREFIX}{TITLE_CORE}\s+(?:in|from|part\s+of)\??\s*$", re.I), "part_of_series"),
            (re.compile(rf"is\s+{MOVIE_PREFIX}{TITLE_CORE}\s+part\s+of\s+a\s+series\??\s*$", re.I), "part_of_series"),
            (re.compile(rf"which\s+franchise\s+is\s+{MOVIE_PREFIX}{TITLE_CORE}\s+in\??\s*$", re.I), "part_of_series"),

            # --- 18. CUSTOM RATING (http://ddis.ch/atai/rating) ---
            # Note: Distinct from MPAA ("rated R"). This captures "score", "grade", "points".
            (re.compile(rf"what\s+is\s+the\s+score\s+(?:of|for)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "custom_rating"),
            (re.compile(rf"what\s+is\s+the\s+rating\s+(?:of|for)\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "custom_rating"), # Ambiguous with MPAA, but common
            (re.compile(rf"how\s+good\s+is\s+{MOVIE_PREFIX}{TITLE_CORE}\??\s*$", re.I), "custom_rating"),
        ]

    def provide_patterns(this): 
        return this.patterns