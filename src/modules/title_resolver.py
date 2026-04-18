import json
import os
PREFIX_WD = "http://www.wikidata.org/entity/"

from modules.paths import ensure_datapath
from modules.title_extraction_latest import extract_titles as extract_titles_latest

class TitleResolver:
    def __init__(self, dict_filename: str = "titles_to_qid.json"):
        """
        Initialize the resolver:
        - Reads DATAPATH env variable
        - Loads the JSON dictionary
        """
        data_path = ensure_datapath()
        dict_path = os.path.join(str(data_path), dict_filename)

        try:
            with open(dict_path, "r", encoding="utf8") as f:
                self.t_q_dict = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dictionary file not found at: {dict_path}")
        
        self.q_t_dict = {v: k for k, v in self.t_q_dict.items()}

    def extract_titles(self, text: str):
        """
        Extract movie titles using the shared staged extractor used by the Latest debugger.
        """
        return extract_titles_latest(text, self)

    def qid_from_title(self, title_list):
        """
        Returns a list of Q-ID's for a movie title list.
        """
        return [PREFIX_WD + self.t_q_dict[t] for t in title_list]

    def title_from_qid(self, qid_list):
        """
        Returns a list of movie titles for a Q-ID list.
        """
        titles = []
        for qid in qid_list:
            # normalize: remove prefix if present
            if isinstance(qid, str) and qid.startswith(PREFIX_WD):
                key = qid[len(PREFIX_WD):]  # "http://.../Q248051" -> "Q248051"
            else:
                key = qid

            title = self.q_t_dict.get(key)
            if title is None:
                # fallback: keep the ID so we don't crash or silently drop it
                continue
            else:
                titles.append(title)

        return titles
