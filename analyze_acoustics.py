import parselmouth
from parselmouth import praat
import pandas as pd
import numpy as np
from glob import glob
import os
import warnings
from tqdm import tqdm
import math


def custom_warning(message, category, filename, lineno, line=None):
    return "{}:{}: {}: {}\n".format(filename, lineno, category.__name__, message)


warnings.formatwarning = custom_warning


class Analyzer:
    """
    Main class, initialization calls included analysis methods, depending on user input.
    """

    def __init__(self) -> None:
        self.directory = input(
            "Input path, containing Praat TextGrids and sound files in sub-folders: "
        )
        self.method_calls = list()

        method_prompts = [
            "Get vowel durations? [Y/n]: ",
            "Get formant averages? [Y/n]: ",
            "Get formant dispersions per speaker? [Y/n]: ",
            "Get RMS values? [Y/n]: ",
            "Get spectral tilt? [Y/n]: ",
            "Get center of gravity? [Y/n]: ",
            "Get word durations? [Y/n]: ",
            "Get target and peak height relative to low end? [Y/n]",
        ]

        for i in range(len(method_prompts)):
            call_method = input(method_prompts[i])

            if call_method in ["", "y", "Y"]:
                self.method_calls.append(True)
            else:
                self.method_calls.append(False)

        self.outfile = input("Output file, CSV to create or append to: ")
        self.speaker_sex = input("CSV Table specifying speaker sex: ")

        if any(self.method_calls):
            self.data = pd.DataFrame(
                columns=["speaker", "utterance", "filepath", "wavpath"]
            )
            self.collection = self.collect_from_directory()

            speaker_sex_path = os.path.abspath(
                os.path.join(self.directory, self.speaker_sex)
            )
            if os.path.isfile(speaker_sex_path):
                self.speaker_sex = pd.read_csv(
                    os.path.join(self.directory, self.speaker_sex)
                )
                self.data = self.data.merge(
                    self.speaker_sex, how="left", on=["speaker"]
                )

            self.data["sound_obj"] = self.data.apply(
                lambda row: parselmouth.Sound(row["wavpath"]), axis=1
            )

            if self.method_calls[0]:
                self.data = self.data.assign(
                    v1_start=np.nan, v1_end=np.nan, v1_duration=np.nan
                )
                self.get_vowel_duration()

            if self.method_calls[1]:
                # TODO: Failsafe if method_calls[0] == False
                self.data = self.data.assign(f1=np.nan, f2=np.nan, f3=np.nan)
                self.get_formants()

            if self.method_calls[2]:

                if not self.method_calls[1]:
                    raise Exception(
                        "Formant dispersion measurement requires formants to be calculated as well."
                    )

                self.data = self.data.assign(
                    f1_f2_dispersion=np.nan, f2_f3_dispersion=np.nan
                )
                self.get_formant_dispersions()

            if self.method_calls[3]:

                if not self.method_calls[0]:
                    raise Exception(
                        "RMS calculation requires vowel durations to be calculated as well."
                    )

                self.data = self.data.assign(v1_rms=np.nan)
                self.get_rms()

            if self.method_calls[4]:

                if not self.method_calls[0]:
                    raise Exception(
                        "Spectral tilt calculation requires vowel durations to be calculated as well."
                    )

                self.data = self.data.assign(v1_obj=None, v1_tilt=np.nan)
                self.get_spectral_tilt()

            if self.method_calls[5]:

                if not self.method_calls[0]:
                    raise Exception(
                        "Spectral center of gravity calculation requires vowel durations to be calculated as well."
                    )

                if self.method_calls[4]:
                    self.data = self.data.assign(v1_cog=np.nan)
                else:
                    self.data = self.data.assign(v1_obj=np.nan, v1_cog=np.nan)

                self.get_center_of_gravity()

            if self.method_calls[6]:

                self.data = self.data.assign(
                    tool_duration=np.nan,
                    target_duration=np.nan,
                    ratio_word_duration=np.nan,
                )
                self.get_word_durations()

            if self.method_calls[7]:

                self.data = self.data.assign(
                    exc_target_low_end=np.nan, exc_peak_low_end=np.nan
                )
                self.get_relative_heights()

            drop_cols = [
                "filepath",
                "wavpath",
                "sound_obj",
                "v1_obj",
                "v1_start",
                "v1_end",
            ]
            for col in drop_cols:
                if col in self.data.columns:
                    self.data = self.data.drop(columns=col, axis=1)

            if os.path.isfile(os.path.join(self.directory, self.outfile)):
                input_df = pd.read_csv(os.path.join(self.directory, self.outfile))
                input_df = input_df.astype({"speaker": "object", "utterance": "int32"})
                self.data = self.data.astype(
                    {"speaker": "object", "utterance": "int32"}
                )

                output_df = input_df.merge(
                    self.data, how="inner", on=["speaker", "utterance"]
                )

                with open(os.path.join(self.directory, self.outfile), "w+") as outfile:
                    output_df.to_csv(outfile, sep=",")

            else:
                with open(os.path.join(self.directory, self.outfile), "w+") as outfile:
                    self.data.to_csv(outfile, sep=",")

        else:
            print("No operations performed. Exiting.")

    def collect_from_directory(self):

        cwd = os.getcwd()

        if self.directory and isinstance(self.directory, str):
            if os.path.exists(os.path.dirname(self.directory)) or os.path.exists(
                os.path.join(cwd, self.directory)
            ):

                dir_content = os.listdir(self.directory)
                collected_items = {
                    session: np.asarray(
                        glob(os.path.join(self.directory, session, "*.TextGrid"))
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

                            self.data.loc[i, ["v1_start", "v1_end", "v1_duration"]] = [
                                v1_start,
                                v1_end,
                                v1_duration,
                            ]

                        else:
                            self.data.loc[i, ["v1_start", "v1_end", "v1_duration"]] = [
                                np.nan,
                                np.nan,
                                np.nan,
                            ]
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
                                i, ["exc_target_low_end", "exc_peak_low_end"]
                            ] = [targ_vs_low, peak_vs_low]
                        else:
                            warnings.warn(
                                "{}-{} is missing word annotations or the word annotations could not be automatically determined.".format(
                                    row["speaker"], row["utterance"]
                                ),
                                UserWarning,
                            )

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


if __name__ == "__main__":
    Analyzer()
