from modules.recommender.recommender import KNNRecommender
from modules.title_resolver import TitleResolver
from utils.message_blocks import debug_block
PREFIX_WD = "http://www.wikidata.org/entity/"
class Recommendation:
    def __init__(self, title_resolver: TitleResolver, rec):
        self.title_resolver = title_resolver
        self.rec = rec

    def de_prefix(self, prefix, string):
        if string.startswith(prefix):
            return string[len(prefix):]
        else:
            return string

    def recommendation_string(self, message: str):
        liked_qid_list = self.title_resolver.qid_from_title(self.title_resolver.extract_titles(message))
        rec_qid_list = self.rec.recommend(liked_qid_list)
        rec_title_list = self.title_resolver.title_from_qid(rec_qid_list)

        return "I recommend " + ", ".join(rec_title_list) + "."

class RecommendationDebugger:
    def __init__(self, rec: Recommendation):
        self.rec = rec

    def print_debug_message(self):
        """
        Prints debug messages for the KNNRecommender class.
        """
        message = "Given that I like The Lion King, Pocahontas, " + \
        "and The Beauty and the Beast, can you recommend some movies?"
        with debug_block():
            print("recommendation:\n", self.rec.recommendation_string(message))

