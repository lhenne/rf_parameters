from os import read
from os.path import basename
import parselmouth
from parselmouth import praat
import pandas as pd
import numpy as np
from glob import glob
import re
import os


def collect_from_directory(directory):

    cwd = os.getcwd()

    if directory and isinstance(directory, str):
        if os.path.exists(os.path.dirname(directory)) or os.path.exists(
            os.path.join(cwd, directory)
        ):

            dir_content = os.listdir(directory)
            collected_items = {
                session: np.asarray(glob(os.path.join(directory, session, "*.TextGrid"))
                )
                for session in dir_content
            }

            return collected_items

        else:
            
            raise ValueError("Please enter a valid path")

    elif directory:
        raise TypeError("Please enter a valid path")

    else:
        raise TypeError("Please enter a valid path")
    