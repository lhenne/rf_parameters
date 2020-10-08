import parselmouth
from parselmouth import praat
import pandas as pd
import numpy as np
from glob import glob
import os
import warnings
from tqdm import tqdm
import math
import json


def custom_warning(message, category, filename, lineno, line=None):
    return "{}:{}: {}: {}\n".format(
        filename, lineno, category.__name__, message
    )  # make warnings more informative for end user


warnings.formatwarning = custom_warning


class Analyzer:
    def __init__(self) -> None:

        with open("config.json", "r") as cfg:
            self.config = json.load(cfg)

        if os.path.exists(self.config["directory"]):
            pass
        else:
            raise ValueError("Please enter a valid path.")

        self.method_calls = [k for k, v in self.config["options"].items() if v]

        if any(self.method_calls):
            self.data = pd.DataFrame(
                columns=["speaker", "utterance", "filepath", "wavpath"]
            )
            self.collection = self.collect_from_directory()
            
            self.data["sound_obj"] = self.data.apply(
                lambda row: parselmouth.Sound(row["wavpath"]), axis=1
            )

            if "vowel duration" in self.method_calls:
                self.data = self.data.assign(
                    v1_start=np.nan, v1_end=np.nan, v1_duration=np.nan, textgrid=np.nan
                )
                self.get_vowel_duration()

            if "formant averages" in self.method_calls:
                self.data = self.data.assign(
                    f1=np.nan, f2=np.nan, f3=np.nan, formant_obj=np.nan
                )
                self.get_formants()

            if "formant dispersions" in self.method_calls:

                if not "formant averages" in self.method_calls:
                    raise Exception(
                        "Formant dispersion measurement requires formants to be calculated as well."
                    )

                self.data = self.data.assign(
                    f1_f2_dispersion=np.nan, f2_f3_dispersion=np.nan
                )
                self.get_formant_dispersions()

            if "rms" in self.method_calls:

                if not "vowel duration" in self.method_calls:
                    raise Exception(
                        "RMS calculation requires vowel durations to be calculated as well."
                    )

                self.data = self.data.assign(v1_rms=np.nan)
                self.get_rms()

            if "spectral tilt" in self.method_calls:

                if not "vowel duration" in self.method_calls:
                    raise Exception(
                        "Spectral tilt calculation requires vowel durations to be calculated as well."
                    )

                self.data = self.data.assign(v1_obj=None, v1_tilt=np.nan)
                self.get_spectral_tilt()

            if "center of gravity" in self.method_calls:

                if not "vowel duration" in self.method_calls:
                    raise Exception(
                        "Spectral center of gravity calculation requires vowel durations to be calculated as well."
                    )

                if "spectral tilt" in self.method_calls:
                    self.data = self.data.assign(v1_cog=np.nan)
                else:
                    self.data = self.data.assign(v1_obj=np.nan, v1_cog=np.nan)

                self.get_center_of_gravity()

            if "word duration" in self.method_calls:

                self.data = self.data.assign(
                    tool_duration=np.nan,
                    target_duration=np.nan,
                    ratio_word_duration=np.nan,
                )
                self.get_word_durations()

            if "relative target and peak height" in self.method_calls:

                self.data = self.data.assign(
                    exc_target_low_end=np.nan, exc_peak_low_end=np.nan, pitch_obj=np.nan
                )
                self.get_relative_heights()

            if "h1-h2" in self.method_calls:

                self.data = self.data.assign(h1_h2=np.nan)
                self.get_h1_h2()

            drop_cols = [
                "filepath",
                "wavpath",
                "sound_obj",
                "textgrid",
                "pitch_obj",
                "formant_obj",
                "v1_obj",
                "v1_start",
                "v1_end",
                "sex",
            ]
            for col in drop_cols:
                if col in self.data.columns:
                    self.data = self.data.drop(columns=col, axis=1)

            in_abspath = os.path.join(
                self.config["directory"], self.config["input_file"]
            )
            out_abspath = os.path.join(
                self.config["directory"], self.config["output_file"]
            )

            if os.path.isfile(in_abspath):
                input_df = pd.read_csv(in_abspath)
                input_df = input_df.astype({"speaker": "object", "utterance": "int32"})
                self.data = self.data.astype(
                    {"speaker": "object", "utterance": "int32"}
                )

                output_df = input_df.merge(
                    self.data, how="inner", on=["speaker", "utterance"]
                )

                with open(out_abspath, "w+") as outfile:
                    output_df.to_csv(outfile, sep=",", index=False)

            else:
                with open(out_abspath, "w+") as outfile:
                    self.data.to_csv(outfile, sep=",", index=False)

        else:
            print("No operations performed. Exiting.")

    def collect_from_directory(self):

        cwd = os.getcwd()

        if self.config["directory"] and isinstance(self.config["directory"], str):
            if os.path.exists(
                os.path.dirname(self.config["directory"])
            ) or os.path.exists(os.path.join(cwd, self.config["directory"])):

                dir_content = os.listdir(self.config["directory"])
                collected_items = {
                    session: np.asarray(
                        glob(
                            os.path.join(
                                self.config["directory"], session, "*.TextGrid"
                            )
                        )
                    )
                    for session in dir_content
                }

                for value in collected_items.values():
                    if isinstance(value, (np.ndarray, np.generic)) or isinstance(
                        value, list
                    ):
                        pass
                    else:
                        raise TypeError(
                            "All values of `collection` must be of type list."
                        )

                self.data["filepath"] = pd.concat(
                    {k: pd.Series(v) for k, v in collected_items.items()}
                )
                self.data["wavpath"] = [
                    "".join([filepath.split(".TextGrid")[0], ".wav"])
                    for filepath in self.data["filepath"]
                ]
                self.data["utterance"] = [
                    filepath.split("/")[-1].split(".TextGrid")[0]
                    for filepath in self.data["filepath"]
                ]

                index_tup = self.data.index
                self.data["speaker"] = [tup[0] for tup in index_tup]
                self.data.reset_index(drop=True, inplace=True)

            else:
                raise ValueError("Please enter a valid path")

        else:
            raise TypeError("Please enter a valid path")

    def get_vowel_duration(self):

        """
        * Import the DataFrame created by `collect_from_directory`
        * Process the TextGrid file with praat-parselmouth
        * Return a DataFrame containing speaker, utterance, TextGrid-filename and vowel duration
        """

        if isinstance(self.data, pd.DataFrame) and "filepath" in self.data.columns:

            for i, row in tqdm(
                self.data.iterrows(),
                desc="Extracting vowel intervals",
                total=len(self.data),
                leave=True,
                position=0,
            ):
                textgrid = parselmouth.Data.read(row["filepath"])

                numtiers = praat.call(textgrid, "Get number of tiers")
                found_vowel_tier = False

                for tiernum in range(1, numtiers + 1):
                    tiername = praat.call(textgrid, "Get tier name", tiernum)

                    if tiername == "Vowel":
                        found_vowel_tier = True
                        num_intervals = praat.call(
                            textgrid, "Get number of intervals", tiernum
                        )

                        if num_intervals == 3:
                            v1_start = praat.call(
                                textgrid, "Get start time of interval", tiernum, 2
                            )
                            v1_end = praat.call(
                                textgrid, "Get end time of interval", tiernum, 2
                            )
                            v1_duration = (v1_end - v1_start) * 1000

                            self.data.loc[
                                i, ["v1_start", "v1_end", "v1_duration", "textgrid"]
                            ] = [v1_start, v1_end, v1_duration, textgrid]

                        else:
                            self.data.loc[
                                i, ["v1_start", "v1_end", "v1_duration", "textgrid"]
                            ] = [np.nan, np.nan, np.nan, textgrid]
                            warnings.warn(
                                "{}-{} is missing a V1 annotation or the V1 annotation could not be automatically determined.".format(
                                    row["speaker"], row["utterance"]
                                ),
                                UserWarning,
                            )

                if found_vowel_tier is False:
                    warnings.warn(
                        "{}-{} does not contain Vowel tier.".format(
                            row["speaker"], row["utterance"]
                        ),
                        UserWarning,
                    )
                else:
                    pass

            self.data.sort_values("utterance")

        else:
            raise TypeError(
                "Please provide a dictionary-type collection of utterances and a pandas DataFrame for the output."
            )

    def get_formants(self):

        """
        * Use the V1 sound object from the output of `get_vowel_duration()` to get the average Hz value of formants F1, F2, F3 in that segment.
        """

        if isinstance(self.data, pd.DataFrame) and "sound_obj" in self.data.columns:

            for i, row in tqdm(
                self.data.iterrows(),
                desc="Extracting V1 formant averages",
                total=len(self.data),
                leave=True,
                position=0,
            ):
                if isinstance(row["sound_obj"], parselmouth.Sound) and not np.isnan(
                    row["v1_start"]
                ):

                    formant_obj = None

                    if "sex" in self.data.columns:
                        if row["sex"] == "m":
                            formant_obj = row["sound_obj"].to_formant_burg(
                                maximum_formant=4500.0
                            )
                        elif row["sex"] == "f":
                            formant_obj = row["sound_obj"].to_formant_burg(
                                maximum_formant=5500.0
                            )

                    else:
                        formant_obj = row["sound_obj"].to_formant_burg(
                            maximum_formant=5500.0
                        )

                    self.data.loc[i, "f1"] = praat.call(
                        formant_obj,
                        "Get mean",
                        1,
                        row["v1_start"],
                        row["v1_end"],
                        "Hertz",
                    )
                    self.data.loc[i, "f2"] = praat.call(
                        formant_obj,
                        "Get mean",
                        2,
                        row["v1_start"],
                        row["v1_end"],
                        "Hertz",
                    )
                    self.data.loc[i, "f3"] = praat.call(
                        formant_obj,
                        "Get mean",
                        3,
                        row["v1_start"],
                        row["v1_end"],
                        "Hertz",
                    )
                    self.data.loc[i, "formant_obj"] = formant_obj

                else:
                    warnings.warn(
                        "Skipping formant measurement for {}-{}: missing V1 segment sound data.".format(
                            row["speaker"], row["utterance"]
                        ),
                        UserWarning,
                    )

            self.data.sort_values("utterance")

        else:
            raise TypeError(
                "Please provide a DataFrame containing a column of V1 segment sound data."
            )

    def get_formant_dispersions(self):

        """
        Use the formant measurements from `get_formants()` to calculate formant dispersion for each speaker
        """

        if isinstance(self.data, pd.DataFrame) and all(
            col in self.data.columns
            for col in ["f1", "f2", "f3", "f1_f2_dispersion", "f2_f3_dispersion"]
        ):
            for speaker in tqdm(
                self.data["speaker"].unique(),
                desc="Calculating formant dispersions for each speaker",
                total=len(self.data["speaker"].unique()),
                leave=True,
                position=0,
            ):
                utterances = self.data.loc[
                    self.data["speaker"] == speaker, ["f1", "f2", "f3"]
                ]
                f1s = list(utterances["f1"].dropna())
                f2s = list(utterances["f2"].dropna())
                f3s = list(utterances["f3"].dropna())

                if not len(f1s) == len(f2s) == len(f3s):
                    warnings.warn(
                        "There are more formants of one kind than of another.",
                        UserWarning,
                    )

                formant_count = max(len(f1s), len(f2s), len(f3s))

                if not formant_count == len(utterances):
                    warnings.warn(
                        "One or more utterances do not have any corresponding formant values."
                    )

                f1_f2_dispersion = sum(
                    [f2s[i] - f1s[i] for i in range(formant_count)]
                ) / (formant_count - 1)
                f2_f3_dispersion = sum(
                    [f3s[i] - f2s[i] for i in range(formant_count)]
                ) / (formant_count - 1)

                self.data.loc[
                    self.data["speaker"] == speaker, "f1_f2_dispersion"
                ] = f1_f2_dispersion
                self.data.loc[
                    self.data["speaker"] == speaker, "f2_f3_dispersion"
                ] = f2_f3_dispersion
        else:
            raise TypeError("Please provide a DataFrame containing formant data.")

    def get_rms(self):

        """
        Calculate the root-mean-square energy over the duration of the vowel.
        """

        if isinstance(self.data, pd.DataFrame) and all(
            col in self.data.columns
            for col in ["v1_rms", "v1_start", "v1_end", "sound_obj"]
        ):
            for i, row in tqdm(
                self.data.iterrows(),
                desc="Calculating vowel RMS",
                total=len(self.data),
                leave=True,
                position=0,
            ):
                if isinstance(row["sound_obj"], parselmouth.Sound):
                    self.data.loc[i, "v1_rms"] = row["sound_obj"].get_root_mean_square(
                        from_time=row["v1_start"], to_time=row["v1_end"]
                    )

                else:
                    warnings.warn(
                        "{}-{} does not contain Vowel tier. NA value inserted.".format(
                            row["speaker"], row["utterance"]
                        ),
                        UserWarning,
                    )

        else:
            raise TypeError("Please provide a DataFrame containing vowel data.")

    def get_spectral_tilt(self):

        """
        Calculate the spectral tilt over the timespan of the V1 label.
        Spectral tilt definition: Mean value of the first Mel-frequency cepstral coefficient (C1).
        To extract this value, a Praat MFCC object has to be calculated.
        """

        if isinstance(self.data, pd.DataFrame) and all(
            col in self.data.columns
            for col in ["sound_obj", "v1_start", "v1_end", "v1_obj", "v1_tilt"]
        ):
            for i, row in tqdm(
                self.data.iterrows(),
                desc="Extracting V1 audio part, creating MFCC object and extracting mean C1.",
                total=len(self.data),
                leave=True,
                position=0,
            ):
                if isinstance(row["sound_obj"], parselmouth.Sound) and not np.isnan(
                    row["v1_start"]
                ):
                    v1_obj = row["sound_obj"].extract_part(
                        from_time=row["v1_start"], to_time=row["v1_end"]
                    )
                    v1_mfcc = v1_obj.to_mfcc(number_of_coefficients=1)
                    v1_tilt = np.mean(v1_mfcc.to_array()[1])

                    self.data.loc[i, "v1_obj"] = v1_obj
                    self.data.loc[i, "v1_tilt"] = v1_tilt
                else:
                    warnings.warn(
                        "{}-{} does not contain Vowel tier. NA value inserted.".format(
                            row["speaker"], row["utterance"]
                        ),
                        UserWarning,
                    )

        else:
            raise TypeError("Please provide a DataFrame containing vowel data.")

    def get_center_of_gravity(self):

        """
        Calculate the center of gravity over the timespan of the V1 label.
        To extract this value, a Praat Spectrum object has to be calculated.
        """

        if isinstance(self.data, pd.DataFrame) and all(
            col in self.data.columns
            for col in ["sound_obj", "v1_start", "v1_end", "v1_obj", "v1_cog"]
        ):
            for i, row in tqdm(
                self.data.iterrows(),
                desc="Creating Spectrum object and extracting center of gravity.",
                total=len(self.data),
                leave=True,
                position=0,
            ):
                if isinstance(row["v1_obj"], parselmouth.Sound):
                    self.data.loc[i, "v1_cog"] = (
                        row["v1_obj"].to_spectrum().get_centre_of_gravity()
                    )

                else:
                    if not np.isnan(row["v1_start"]):
                        v1_obj = row["sound_obj"].extract_part(
                            from_time=row["v1_start"], to_time=row["v1_end"]
                        )
                        self.data.loc[
                            i, "v1_cog"
                        ] = v1_obj.to_spectrum().get_centre_of_gravity()

                    else:
                        warnings.warn(
                            "{}-{} does not contain Vowel tier. NA value inserted.".format(
                                row["speaker"], row["utterance"]
                            ),
                            UserWarning,
                        )

        else:
            raise TypeError("Please provide a DataFrame containing vowel data.")

    def get_word_durations(self):

        """
        Get the durations of tool and target word labels, and then calculate the ratio of tool to target.
        """

        if isinstance(self.data, pd.DataFrame) and all(
            col in self.data.columns
            for col in [
                "filepath",
                "tool_duration",
                "target_duration",
                "ratio_word_duration",
            ]
        ):
            for i, row in tqdm(
                self.data.iterrows(),
                desc="Calculating word durations and ratio of words durations.",
                total=len(self.data),
                leave=True,
                position=0,
            ):
                textgrid = parselmouth.Data.read(row["filepath"])

                numtiers = praat.call(textgrid, "Get number of tiers")
                found_word_tier = False

                for tiernum in range(1, numtiers + 1):
                    tiername = praat.call(textgrid, "Get tier name", tiernum)

                    if tiername == "Word":
                        found_word_tier = True
                        num_intervals = praat.call(
                            textgrid, "Get number of intervals", tiernum
                        )

                        if num_intervals == 5:
                            tool_start = praat.call(
                                textgrid, "Get start time of interval", tiernum, 2
                            )
                            tool_end = praat.call(
                                textgrid, "Get end time of interval", tiernum, 2
                            )
                            tool_duration = (tool_end - tool_start) * 1000

                            target_start = praat.call(
                                textgrid, "Get start time of interval", tiernum, 4
                            )
                            target_end = praat.call(
                                textgrid, "Get end time of interval", tiernum, 4
                            )
                            target_duration = (target_end - target_start) * 1000

                            ratio_word_duration = tool_duration / target_duration

                            self.data.loc[
                                i,
                                [
                                    "tool_duration",
                                    "target_duration",
                                    "ratio_word_duration",
                                ],
                            ] = [tool_duration, target_duration, ratio_word_duration]

                        else:
                            warnings.warn(
                                "{}-{} is missing word annotations or the word annotations could not be automatically determined.".format(
                                    row["speaker"], row["utterance"]
                                ),
                                UserWarning,
                            )

                if found_word_tier is False:
                    warnings.warn(
                        "{}-{} does not contain Word tier.".format(
                            row["speaker"], row["utterance"]
                        ),
                        UserWarning,
                    )
                else:
                    pass

        else:
            raise TypeError(
                "Please provide a DataFrame containing the necessary columns."
            )

    def get_relative_heights(self):

        """
        Calculate the relative heights for target labels (second tone label) and peak labels ("H" tone label), compared to the low end ("L%")
        """

        if isinstance(self.data, pd.DataFrame) and all(
            col in self.data.columns
            for col in [
                "filepath",
                "sound_obj",
                "exc_target_low_end",
                "exc_peak_low_end",
            ]
        ):
            for i, row in tqdm(
                self.data.iterrows(),
                desc="Calculating relative target and peak heights.",
                total=len(self.data),
                leave=True,
                position=0,
            ):
                textgrid = parselmouth.Data.read(row["filepath"])

                numtiers = praat.call(textgrid, "Get number of tiers")
                found_tone_tier = False

                for tiernum in range(1, numtiers + 1):
                    tiername = praat.call(textgrid, "Get tier name", tiernum)

                    if tiername == "f0":
                        found_tone_tier = True
                        num_labels = praat.call(
                            textgrid, "Get number of points", tiernum
                        )

                        if num_labels == 3:
                            pitch_obj = row["sound_obj"].to_pitch(
                                pitch_floor=50, pitch_ceiling=500
                            )

                            labels = [
                                praat.call(
                                    textgrid, "Get label of point", tiernum, point
                                )
                                for point in range(1, num_labels + 1)
                            ]

                            timestamps = [
                                praat.call(
                                    textgrid, "Get time of point", tiernum, point
                                )
                                for point in range(1, num_labels + 1)
                            ]

                            f0s = [
                                pitch_obj.get_value_at_time(timestamp)
                                for timestamp in timestamps
                            ]

                            targ_vs_low = 12 * math.log2(f0s[1] / f0s[2])

                            peak = labels.index("H")
                            peak_vs_low = 12 * math.log2(f0s[peak] / f0s[2])

                            self.data.loc[
                                i,
                                ["exc_target_low_end", "exc_peak_low_end", "pitch_obj"],
                            ] = [targ_vs_low, peak_vs_low, pitch_obj]
                        else:
                            pass

                if found_tone_tier is False:
                    warnings.warn(
                        "{}-{} does not contain tone tier.".format(
                            row["speaker"], row["utterance"]
                        ),
                        UserWarning,
                    )
                else:
                    pass

        else:
            raise TypeError(
                "Please provide a DataFrame containing the necessary columns."
            )

    def get_h1_h2(self):

        """
        Calculate H1-H2
        """

        for i, row in tqdm(
            self.data.iterrows(),
            desc="Calculating H1-H2.",
            total=len(self.data),
            leave=True,
            position=0,
        ):

            if "pitch_obj" in row.index and pd.notnull(row["pitch_obj"]):
                pitch_obj = row["pitch_obj"]
            else:
                pitch_obj = row["sound_obj"].to_pitch(pitch_floor=50, pitch_ceiling=500)

            snd_obj = row["sound_obj"]

            if pd.notnull(row["v1_start"]):
                v1_start = row["v1_start"]
                v1_end = row["v1_end"]

                q25 = 0.75 * praat.call(
                    pitch_obj, "Get quantile", v1_start, v1_end, 0.25, "Hertz"
                )
                q75 = 2.5 * praat.call(
                    pitch_obj, "Get quantile", v1_start, v1_end, 0.75, "Hertz"
                )

                try:
                    v1_obj = snd_obj.extract_part(from_time=v1_start, to_time=v1_end)
                    pitch_obj_2 = v1_obj.to_pitch_cc(pitch_floor=q25, pitch_ceiling=q75)
                    pp_obj = praat.call(pitch_obj_2, "To PointProcess")
                    ltas_obj = praat.call([v1_obj, pp_obj], "To Ltas (only harmonics)", 20, 0.0001, 0.02, 1.3)
                
                    h1 = praat.call(ltas_obj, "Get value in bin", 2)
                    h2 = praat.call(ltas_obj, "Get value in bin", 3)
                    
                    h1_h2 = h1 - h2
                    
                    self.data.loc[i, "h1_h2"] = [
                        h1_h2
                    ]
                except:
                    continue

               

            else:
                self.data.loc[i, "h1_h2"] = [
                    np.nan,
                ]

if __name__ == "__main__":
    Analyzer()
