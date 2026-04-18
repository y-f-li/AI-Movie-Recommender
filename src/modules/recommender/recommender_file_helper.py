import pandas as pd
from pathlib import Path
from utils.message_blocks import debug_block

class RatingsFileProcessor:
    """
    This class handles processing the vanilla ratings files to merge
    them and edit them into a new radius file that includes centered ratings.
    """
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.new_ratings_path = Path(dataset_path + "new_ratings.csv")
        self.user_ratings = None
        self.item_ratings = None
        self.new_ratings = None

    def read_files(self):
        """
        Read the recommender rating files to prepare for the creation of the new ratings file.
        """
        self.user_ratings = pd.read_csv(self.dataset_path / "user_ratings.csv")
        self.item_ratings = pd.read_csv(self.dataset_path / "item_ratings.csv")
        self.item_ratings = self.item_ratings.rename(columns={"rating": "global_rating"})

    def new_ratings_handler(self):
        """
        Generate centered ratings as the Difference between global ratings and user ratings. 
        Save the data to a new column in a new file . 
        The file is called new_ratings.Csv The column is named centered_rating. 
        Write the csv file new_ratings.csv into the dataset folder.
        """
        self.new_ratings = self.user_ratings.merge(
                     self.item_ratings,
                     on="item_id",   # match rows by item_id
                     how="left"      # keep all rows from user_ratings
                    )
        self.new_ratings["centered_rating"] = self.new_ratings["rating"] - self.new_ratings["global_rating"]
        self.new_ratings.to_csv(self.new_ratings_path, index=False)

    def recommender_file_prep(self):
        """
        Check if the new ratings file is already in the dataset folder. 
        If it isn't, create it.
        """
        if self.new_ratings_path.exists():
            self.new_ratings = pd.read_csv(self.new_ratings_path)
        else:
            self.read_files()
            self.new_ratings_handler()

class RatingsFileProcessorDebugger:
    """
    This class is a debug helper for rating preprocessor.
    """
    def __init__(self, rp: RatingsFileProcessor):
        self.rp = rp

    def print_debug_messages(self):
        """
        Prints debug messages for the rating preprocessor class.
        """
        with debug_block():
            print(f"new_rating.csv exists: {self.rp.new_ratings_path.exists()}")
            print(f"head of new_ratings.csv:\n{self.rp.new_ratings.head()}")
