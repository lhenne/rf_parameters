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
    
    
def get_vowel_duration(collection, output_df):
    
    """
    * Import the dictionary created by `collect_from_directory`
    * Process the TextGrid file with praat-parselmouth
    * Return a pandas DataFrame containing speaker, utterance, TextGrid-filename and vowel duration
    """
    
    if isinstance(collection, dict) and "filepath" in output_df.columns:
        
        for value in collection.values():
            if isinstance(value, (np.ndarray, np.generic)):
                pass
            else:
                raise TypeError("All values of `collection` must be of type list.")
        
        output_df["filepath"] = pd.concat({k: pd.Series(v) for k, v in collection.items()})
        output_df["recording"] = [filepath.split("/")[-1].split(".TextGrid")[0] for filepath in output_df["filepath"]]
        
        index_tup = output_df.index
        output_df["speaker"] = [tup[0] for tup in index_tup]
        output_df.reset_index(drop = True, inplace = True)
        
        for index, row in output_df.iterrows():
            print("Extracting syllable intervals from... {}-{}".format(row["speaker"], row["filepath"]))
            textgrid = parselmouth.Data.read(row["filepath"])
            
            numtiers = praat.call(textgrid, "Get number of tiers")
            
            for tiernum in range(1, numtiers+1):
                tiername = praat.call(textgrid, "Get tier name", tiernum)
                
                if tiername == "Vowel":
                    numintervals = praat.call(textgrid, "Get number of intervals", tiernum)
                    
                    if numintervals == 3:
                        print(praat.call(textgrid, "Get end time of interval", tiernum, 2))
                        print(praat.call(textgrid, "Get start time of interval", tiernum, 2))
                        row["v1_duration"] = (praat.call(textgrid, "Get end time of interval", tiernum, 2) - praat.call(textgrid, "Get start time of interval", tiernum, 2)) * 1000
                        
            
        return output_df.sort_values("recording")
    
    else:
        raise TypeError("Please provide a dictionary-type collection of recordings and a pandas DataFrame for the output.")
    