from os.path import basename
import parselmouth
from parselmouth import praat
import numpy as np
from glob import glob
import os


def collect_from_directory(directory):

    cwd = os.getcwd()

    if directory and isinstance(directory, str):
        if os.path.exists(os.path.dirname(directory)) or os.path.exists(
            os.path.join(cwd, directory)
        ):

            dir_content = os.listdir(directory)
            collected_items = {
                session: [
                    os.path.basename(file)
                    for file in glob(os.path.join(directory, session, "*.TextGrid"))
                ]
                for session in dir_content
            }

            return collected_items

        else:
            print("Please enter a valid path")
            raise ValueError

    elif directory:
        print("Please enter a valid path")
        raise TypeError

    else:
        print("Please enter a valid path")
        raise TypeError

