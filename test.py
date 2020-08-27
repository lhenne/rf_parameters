import unittest

from numpy.lib.function_base import place
from analyze_acoustics import *

class CollectFromDirectoryTests(unittest.TestCase):
    
    def test_invalid_type_input(self):
        """
        Do invalid type inputs return an error?
        """
        
        for test_case in (20, None, "", True):
            with self.subTest(test_case):
                with self.assertRaises(TypeError):
                    collect_from_directory(test_case)
                    
    def test_bad_string_input(self):
        """
        Do nonexistent paths as input return an error?
        """
        
        for test_case in ("NotAValidPath", "another/kind/of/invalid", "also-2020ü&not_valid/"):
            with self.subTest(test_case):
                with self.assertRaises(ValueError):
                    collect_from_directory(test_case)
        
    
    def test_correct_output_type(self):
        """
        Is the method output a dictionary?
        """
        
        collection = collect_from_directory("test_material/")
        self.assertIsInstance(collection, dict)
        
    def test_correct_output_content(self):
        """
        Does example input return the correct output?
        """
        
        collection = collect_from_directory("test_material/")
        self.assertIn("AH_95", collection.keys())
        
        for test_case in ("test_material/AH_95/0018.TextGrid", "test_material/AH_95/0098.TextGrid", "test_material/AH_95/0057.TextGrid"):
            with self.subTest(test_case):
                self.assertIn(test_case, collection["AH_95"])
                
    def test_file_types_filtered(self):
        """
        Does the output dictionary only contain TextGrid files?
        """
        
        collection = collect_from_directory("test_material/")
        for test_case in ("0018.wav", "0098.wav", "readme.txt"):
            with self.subTest(test_case):
                self.assertNotIn(test_case, collection["AH_95"])
    
    
class GetVowelDurationTests(unittest.TestCase):
    
    def test_invalid_type_input(self):
        """
        Do inputs of invalid data type return an error?
        """
        
        output_df = pd.DataFrame(columns = ["speaker", "recording", "filepath", "wavpath", "v1_wav", "v1_start", "v1_end", "v1_duration"])
        
        for test_case in (None, True, ["AH_95",  ["0018.TextGrid"]], "AH_95: 0018.TextGrid"):
            with self.subTest(test_case):
                with self.assertRaises(TypeError):
                    get_vowel_duration(test_case, output_df)
    
    def test_bad_structure_input(self):
        """
        Do dictionary inputs with wrong structure return an error?
        """
        
        output_df = pd.DataFrame(columns = ["speaker", "recording", "filepath", "wavpath", "v1_wav", "v1_start", "v1_end", "v1_duration"])
        
        for test_case in ({"AH_95": "0018.TextGrid", "AH_95": "0019.TextGrid"}, {("0018.TextGrid", "0019.TextGrid", "0047.TextGrid"): "AH_95"}):
            with self.subTest(test_case):
                with self.assertRaises(TypeError):
                    get_vowel_duration(test_case, output_df)
    
    def test_dataframe_output_type(self):
        """
        Does the function output a pandas.DataFrame object?
        """
    
        collection = collect_from_directory("test_material/")
        output_df = pd.DataFrame(columns = ["speaker", "recording", "filepath", "wavpath", "v1_wav", "v1_start", "v1_end", "v1_duration"])
        output_df = get_vowel_duration(collection, output_df)
            
        self.assertIsInstance(output_df, pd.DataFrame)
    
    def test_dataframe_output_values(self):
        """
        Does the function extract the correct values?
        """
    
        collection = collect_from_directory("test_material/")
        output_df = pd.DataFrame(columns = ["speaker", "recording", "filepath", "wavpath", "v1_wav", "v1_start", "v1_end", "v1_duration"])
        output_df = get_vowel_duration(collection, output_df)
        
        value_0018 = output_df.loc[output_df["recording"] == "0018", "v1_duration"].item()  
        value_0038 = output_df.loc[output_df["recording"] == "0038", "v1_duration"].item()
        value_0058 = output_df.loc[output_df["recording"] == "0058", "v1_duration"].item()  
        
        self.assertAlmostEqual(value_0018, 139.32, places = 1)
        self.assertAlmostEqual(value_0038, 223.54, places = 1)
        self.assertAlmostEqual(value_0058, 128.77, places = 1)
        
    def test_missing_label(self):
        """
        Does the function correctly deal with the one missing V1 label?
        """
        
        collection = collect_from_directory("test_material/")
        output_df = pd.DataFrame(columns = ["speaker", "recording", "filepath", "wavpath", "v1_wav", "v1_start", "v1_end", "v1_duration"])

        with self.assertWarns(UserWarning):
            get_vowel_duration(collection, output_df)

