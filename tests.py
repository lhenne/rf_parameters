import unittest
from analyze_acoustics import *

class AnalyzeAcousticsTests(unittest.TestCase):
    def test_format_data(self):
        collection = collect_from_directory("test_material/")
        self.assertIsInstance(collection, dict)
        self.assertIn("AH_95", collection.keys())
        self.assertIn("0018.TextGrid", collection["AH_95"])
        self.assertIn("0098.TextGrid", collection(["AH_95"]))
        self.assertIn("0057.TextGrid", collection["AH_95"])
        self.assertNotIn("0018.wav", collection["AH_95"])