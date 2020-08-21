import unittest
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
        
        for test_case in ("0018.TextGrid", "0098.TextGrid", "0057.TextGrid"):
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
    