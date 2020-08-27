import parselmouth
from parselmouth import praat
import pandas as pd
import numpy as np
from glob import glob
import os
import warnings
from tqdm import tqdm


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
        output_df["wavpath"] = ["".join([filepath.split(".TextGrid")[0], ".wav"]) for filepath in output_df["filepath"]]
        output_df["recording"] = [filepath.split("/")[-1].split(".TextGrid")[0] for filepath in output_df["filepath"]]
        
        index_tup = output_df.index
        output_df["speaker"] = [tup[0] for tup in index_tup]
        output_df.reset_index(drop = True, inplace = True)
        
        for index, row in tqdm(output_df.iterrows(), desc="Extracting vowel intervals"):
            textgrid = parselmouth.Data.read(row["filepath"])
            
            numtiers = praat.call(textgrid, "Get number of tiers")
            found_vowel_tier = False
            
            for tiernum in range(1, numtiers+1):
                tiername = praat.call(textgrid, "Get tier name", tiernum)
                
                if tiername == "Vowel":
                    found_vowel_tier = True
                    numintervals = praat.call(textgrid, "Get number of intervals", tiernum)
                    
                    if numintervals == 3:
                        row["v1_start"] = praat.call(textgrid, "Get start time of interval", tiernum, 2)
                        row["v1_end"] =praat.call(textgrid, "Get end time of interval", tiernum, 2)
                        row["v1_duration"] = (row["v1_end"] - row["v1_start"]) * 1000
                        
                        sound_obj = parselmouth.Sound(row["wavpath"])
                        row["v1_wav"] = sound_obj.extract_part(row["v1_start"], row["v1_end"])

                    else:
                        row[["v1_start", "v1_end", "v1_duration","v1_wav"]] = [np.nan, np.nan, np.nan, np.nan]
                        warnings.warn("{}-{} is missing a V1 annotation or the V1 annotation could not be automatically determined.".format(row["speaker"], row["recording"]), UserWarning)
            
            if found_vowel_tier is False:
                warnings.warn("{}-{} does not contain Vowel tier.".format(row["speaker"], row["recording"]), UserWarning)
            else:
                pass
            
        return output_df.sort_values("recording")
    
    else:
        raise TypeError("Please provide a dictionary-type collection of recordings and a pandas DataFrame for the output.")
    