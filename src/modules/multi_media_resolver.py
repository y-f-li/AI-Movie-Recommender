import json
import os
import re
import random as rnd
import unicodedata

from modules.paths import ensure_datapath

MIN_TITLE_LEN = 3  # minimum length of title to consider
PREFIX_WD = "http://www.wikidata.org/entity/"

class MultiMediaResolver:
    def __init__(self, human_to_IMDB: str = "human_to_IMDB.json", movie_to_IMDB: str = "movie_to_IMDB.json", images_mapper: str = "images.json"):
        """
        Initialize the resolver:
        - Reads DATAPATH env variable
        - Loads the JSON sources
        """
        data_path = ensure_datapath()
        full_path = os.path.join(str(data_path), "multimedia")

        path_human = os.path.join(full_path, human_to_IMDB)
        path_movie = os.path.join(full_path, movie_to_IMDB)
        path_images = os.path.join(full_path, images_mapper)

        try:
            with open(path_human, "r", encoding="utf8") as f:
                self.human_to_IMDB = json.load(f)
            with open(path_images, "r", encoding="utf8") as f:
                self.images_mapper = json.load(f)
            with open(path_movie, "r", encoding="utf8") as f:
                self.movie_to_IMDB = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Multimedia JSON file(s) not found at: {full_path}")
        

    def answer_multimedia_query(self, query: str):
        #First look up humans within userquery
        human_names = self.extract_keys_within_substring_of_dict(query, self.human_to_IMDB, None)
        imdb_ids_human = self.human_readable_to_IMDBID(human_names, self.human_to_IMDB)
        human_names_name_to_pictureURI = self.build_profile_image_dict_by_name(imdb_ids_human, human_names)


        movie_names = self.extract_keys_within_substring_of_dict(query, self.movie_to_IMDB, human_names)
        imdb_ids_movies = self.human_readable_to_IMDBID(movie_names, self.movie_to_IMDB)
        movie_names_to_posterURI, movie_names_to_backdropsURI = self.build_movie_image_dicts(imdb_ids_movies, movie_names)


        """
        print("=== MULTIMEDIA QUERY DEBUG ===")
        print(f"Humans found         : {human_names}")
        print(f"Human IMDb IDs       : {imdb_ids_human}")
        print(f"Human images         : {human_names_name_to_pictureURI}\n")

        print(f"Movies found         : {movie_names}")
        print(f"Movie IMDb IDs       : {imdb_ids_movies}")
        print(f"Movie images  poster       : {movie_names_to_posterURI}")
        print(f"Movie images  backdrops       : {movie_names_to_backdropsURI}")

        print("================================")
        """


        return self.create_response_string( human_names_name_to_pictureURI, movie_names_to_backdropsURI, movie_names_to_posterURI)

    def create_response_string(self, names_to_URI, movie_to_backdrop_URI, movie_to_posterURI): 
        
        title = "Here are your requested multimedia resources: \n"
        humans = ""
        for human, URIs in names_to_URI.items():
            if URIs :
                humans +=  str(min(len(URIs), 2)) + f" random pictures of {human}: \n"
                for image in rnd.sample(URIs, min(len(URIs), 2)) : humans += "image:"+image + "\n"


        movies_frames = ""
        for movie, URIs in movie_to_backdrop_URI.items():
            if URIs :
                movies_frames +=  str(min(len(URIs), 2)) + f" random backdrops of {movie}: \n"
                for image in rnd.sample(URIs, min(len(URIs), 2)) : movies_frames += "image:"+image + "\n"  
   
        
        movies_poster = ""
        for movie, URIs in movie_to_posterURI.items():
            if URIs :
                movies_poster +=  str(min(len(URIs), 2)) + f" random posters of {movie}: \n"
                for image in rnd.sample(URIs, min(len(URIs), 2)) : movies_poster += "image:"+image + "\n" 

        if(not (humans or movies_frames or movies_poster)): 
            return "Sorry, I the requested multimedia resource cannot be accessed due to it not being present in the provided Knowledge base. "
        answer = title + humans + movies_frames + movies_poster
        print(f"ANSWER string : {answer}")
        return answer

    def build_movie_image_dicts(self, imdb_ids, movie_names):
        posters = {name: [] for name in movie_names}
        backdrops = {name: [] for name in movie_names}
        # lookup table
        id_to_name = dict(zip(imdb_ids, movie_names))
        for entry in self.images_mapper:
            img_type = entry.get("type")
            if img_type == "profile":  
                continue
            img = entry.get("img")
            if not img:
                continue
            img_no_ext = img.rsplit(".", 1)[0]
            movie_list = entry.get("movie", [])
            for pid in imdb_ids:
                if pid in movie_list:
                    name = id_to_name[pid]
                    if img_type == "poster":
                        posters[name].append(img_no_ext)
                    elif img_type == "backdrop":
                        backdrops[name].append(img_no_ext)
        return posters, backdrops
    
    def build_profile_image_dict_by_name(self, imdb_ids, human_names):
        result = {name: [] for name in human_names}
        id_to_name = dict(zip(imdb_ids, human_names))
        for entry in self.images_mapper:
            if entry.get("type") != "profile":
                continue
            img = entry.get("img")
            if not img:
                continue
            img_no_ext = img.rsplit(".", 1)[0]
            cast_list = entry.get("cast", [])
            for pid in imdb_ids:
                if pid in cast_list:
                    name = id_to_name[pid]
                    result[name].append(img_no_ext)

        return result        
    
    def extract_keys_within_substring_of_dict(self, userquery: str, dictionary, list_with_names_already_identified):
        """
        Extract human names found as substrings inside the text.
        Case-insensitive.
        """
        text_lower = self._normalize(userquery)
        found = []
        # Check longest names first
        for name in sorted(dictionary, key=len, reverse=True):
            normalized_name = self._normalize(name)
            pattern = r"\b" + re.escape(normalized_name) + r"\b"
            # Skip very short names
            if len(name) < MIN_TITLE_LEN:
                continue
            if not re.search(pattern, text_lower):
                continue
            if any(normalized_name in self._normalize(longer) for longer in found):
                continue
            if list_with_names_already_identified and any( normalized_name in self._normalize(existing)
                for existing in list_with_names_already_identified):
                continue
            # Word boundary match (safer than plain substring)
            if name != "you":
                found.append(name)
        return found

    def human_readable_to_IMDBID(self, key_list_human_readable, dictionary):
        """
        Returns a list of IMDB ID's for a human name list.
        """
        return [dictionary[t] for t in key_list_human_readable if t in dictionary]
    
    def _normalize(self, s: str) -> str:
        return unicodedata.normalize("NFC", s).lower()


    