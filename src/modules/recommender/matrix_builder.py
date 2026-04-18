from modules.recommender.recommender_file_helper import RatingsFileProcessor
from utils.message_blocks import debug_block
import numpy as np
import pandas as pd

class RatingMatrixBuilder:
    """
    
    """
    def __init__(self, rfp: RatingsFileProcessor):
        self.rfp = rfp

        self.rating_matrix = None

    def build_rating_matrix(self):
        """
        Build a US rating matrix using the new rating file.
        Each row will be a user ID, Each column will be a movie ID, and
        the values will be the centered rating zero if missing.
        """
        if self.rfp.new_ratings is None:
            self.rfp.recommender_file_prep()
        
        self.rating_matrix = self.rfp.new_ratings.pivot_table(
            index="item_id",
            columns="user_id",
            values="centered_rating",
            fill_value=0.0
        )

class RatingMatrixBuilderDebugger:
    """
    This class is a debug helper for RatingMatrixBuilder.
    """
    def __init__(self, rmb: RatingMatrixBuilder):
        self.rmb = rmb

    def print_debug_message(self):
        """
        Prints debug messages for the RatingMatrixBuilder class.
        """
        with debug_block():
            print("ratings matrix shape: ", self.rmb.rating_matrix.shape)
            print(f"head of rating matrix:\n{self.rmb.rating_matrix}" )
