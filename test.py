import unittest
from analyze_acoustics import *

class AnalyzeAcousticsTests(unittest.TestCase):
    
    def test_collect_from_directory(self):
        
        with self.assertRaises(TypeError):
            collect_from_directory(20)
        with self.assertRaises(TypeError):
            collect_from_directory(None)
        with self.assertRaises(TypeError):
            collect_from_directory("")
        with self.assertRaises(ValueError):
            collect_from_directory("NotAValidPath")
        
        collection = collect_from_directory("test_material/")
        self.assertIsInstance(collection, dict)
        self.assertIn("AH_95", collection.keys())
        self.assertIn("0018.TextGrid", collection["AH_95"])
        self.assertIn("0098.TextGrid", collection(["AH_95"]))
        self.assertIn("0057.TextGrid", collection["AH_95"])
        self.assertNotIn("0018.wav", collection["AH_95"])
