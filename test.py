import unittest
from unittest import mock
from unittest.mock import Mock

from numpy.lib.function_base import place
from analyze_acoustics import *

class InputTests(unittest.TestCase):
    
    def test_invalid_path(self):
        """
        Does entering an invalid path (which doesn't exist) return an error?
        """
        
        mock_input = Mock()
        mock_input.side_effect = ["NotAValidPath/", "y", "n", "n", "n", "n", "n", "n", "test_material/everything.csv", "test_material/sex.csv"]
        
        with self.assertRaises(ValueError):
            with mock.patch("builtins.input", mock_input):
                Analyzer()
            
    def test_get_directory(self):
        """
        Is user input for input directory stored correctly?
        """
        
        mock_input = Mock()
        mock_input.side_effect = ["test_material/", "y", "n", "n", "n", "n", "n", "n", "n", "n", "everything.csv", "sex.csv"]
        
        with mock.patch("builtins.input", mock_input):
            tester = Analyzer()
        
        self.assertTrue("test_material/" == tester.directory)
    
    def test_get_output_file(self):
        """
        Is user input for output file stored correctly?
        """
       
        mock_input = Mock()
        mock_input.side_effect = ["test_material/", "y", "n", "n", "n", "n", "n", "n", "n", "n", "everything.csv", "sex.csv"]
        
        with mock.patch("builtins.input", mock_input):
            tester = Analyzer()
        
        self.assertTrue("everything.csv" == tester.outfile)
    
    def test_get_speaker_sex_file(self):
        """
        Is user input for file containing speaker sex information stored correctly?
        """
        
        mock_input = Mock()
        mock_input.side_effect = ["test_material/", "n", "n", "n", "n", "n", "n", "n", "n", "n", "everything.csv", "sex.csv"]
        
        with mock.patch("builtins.input", mock_input):
            tester = Analyzer()
        
        self.assertTrue("sex.csv" == tester.speaker_sex)
            
    
class AnalysisTests(unittest.TestCase):
    
    def test_get_vowel_duration(self):
        """
        Are vowel durations extracted correctly?
        """
        
        mock_input = Mock()
        mock_input.side_effect = ["test_material/", "y", "n", "n", "n", "n", "n", "n", "n", "n", "everything.csv", "sex.csv"]
        
        with mock.patch("builtins.input", mock_input):
            tester = Analyzer()
            
            value_0018 = tester.data.loc[tester.data["utterance"] == 18, "v1_duration"].item()                        
            value_0038 = tester.data.loc[tester.data["utterance"] == 38, "v1_duration"].item()
            value_0058 = tester.data.loc[tester.data["utterance"] == 58, "v1_duration"].item()
            
            self.assertAlmostEqual(value_0018, 139.32, places = 1)
            self.assertAlmostEqual(value_0038, 223.54, places = 1)
            self.assertAlmostEqual(value_0058, 128.77, places = 1)
            
    def test_get_f1(self):
        """
        Are formant averages extracted correctly?
        """
        
        mock_input = Mock()
        mock_input.side_effect = ["test_material/", "y", "y", "n", "n", "n", "n", "n", "n", "n", "everything.csv", "sex.csv"]
        
        with mock.patch("builtins.input", mock_input):
            tester = Analyzer()
            
            value_0019 = tester.data.loc[tester.data["utterance"] == 19, "f1"].item()                        
            value_0039 = tester.data.loc[tester.data["utterance"] == 39, "f1"].item()
            value_0059 = tester.data.loc[tester.data["utterance"] == 59, "f1"].item()
            
            self.assertAlmostEqual(value_0019, 674.983, places = 1)
            self.assertAlmostEqual(value_0039, 364.388, places = 1)
            self.assertAlmostEqual(value_0059, 663.610, places = 1)
            
    def test_get_f2(self):
        """
        Are formant averages extracted correctly?
        """
        
        mock_input = Mock()
        mock_input.side_effect = ["test_material/", "y", "y", "n", "n", "n", "n", "n", "n", "n", "everything.csv", "sex.csv"]
        
        with mock.patch("builtins.input", mock_input):
            tester = Analyzer()
            
            value_0019 = tester.data.loc[tester.data["utterance"] == 19, "f2"].item()                        
            value_0039 = tester.data.loc[tester.data["utterance"] == 39, "f2"].item()
            value_0059 = tester.data.loc[tester.data["utterance"] == 59, "f2"].item()
            
            self.assertAlmostEqual(value_0019, 1143.153, places = 1)
            self.assertAlmostEqual(value_0039, 781.083, places = 1)
            self.assertAlmostEqual(value_0059, 1291.016, places = 1)
        
    def test_get_f3(self):
        """
        Are formant averages extracted correctly?
        """
        
        mock_input = Mock()
        mock_input.side_effect = ["test_material/", "y", "y", "n", "n", "n", "n", "n", "n", "n", "everything.csv", "sex.csv"]
        
        with mock.patch("builtins.input", mock_input):
            tester = Analyzer()
            
            value_0019 = tester.data.loc[tester.data["utterance"] == 19, "f3"].item()                        
            value_0039 = tester.data.loc[tester.data["utterance"] == 39, "f3"].item()
            value_0059 = tester.data.loc[tester.data["utterance"] == 59, "f3"].item()
            
            self.assertAlmostEqual(value_0019, 2420.693, places = 1)
            self.assertAlmostEqual(value_0039, 2616.461, places = 1)
            self.assertAlmostEqual(value_0059, 2485.847, places = 1)
            
    def test_get_f1_f2_dispersion(self):
        """
        Are formant dispersions extracted correctly?
        """
        
        mock_input = Mock()
        mock_input.side_effect = ["test_material/", "y", "y", "y", "n", "n", "n", "n", "n", "n", "everything.csv", "sex.csv"]
        
        with mock.patch("builtins.input", mock_input):
            tester = Analyzer()
            
            value_0018 = tester.data.loc[tester.data["utterance"] == 18, "f1_f2_dispersion"].item()                        
            
            self.assertAlmostEqual(value_0018, 499.455, places = 1)
            
    def test_get_f2_f3_dispersion(self):
        """
        Are formant dispersions extracted correctly?
        """
        
        mock_input = Mock()
        mock_input.side_effect = ["test_material/", "y", "y", "y", "n", "n", "n", "n", "n", "n", "everything.csv", "sex.csv"]
        
        with mock.patch("builtins.input", mock_input):
            tester = Analyzer()
            
            value_0018 = tester.data.loc[tester.data["utterance"] == 18, "f2_f3_dispersion"].item()                      
            
            self.assertAlmostEqual(value_0018, 1489.802, places = 1)
        

    def test_get_rms(self):
        """
        Are RMS values extracted correctly?
        """
        
        mock_input = Mock()
        mock_input.side_effect = ["test_material/", "y", "n", "n", "y", "n", "n", "n", "n", "n", "everything.csv", "sex.csv"]
        
        with mock.patch("builtins.input", mock_input):
            tester = Analyzer()
            
            value_0025 = tester.data.loc[tester.data["utterance"] == 25, "v1_rms"].item()                      
            value_0033 = tester.data.loc[tester.data["utterance"] == 33, "v1_rms"].item()  
            value_0061 = tester.data.loc[tester.data["utterance"] == 61, "v1_rms"].item()  
            
            self.assertAlmostEqual(value_0025, 0.09407199488169092, places = 3)
            self.assertAlmostEqual(value_0033, 0.1048775907277835, places = 3)
            self.assertAlmostEqual(value_0061, 0.10630864064829092, places = 3)
            
    def test_get_spectral_tilt(self):
        """
        Are spectral tilt values extracted correctly?
        """
        
        mock_input = Mock()
        mock_input.side_effect = ["test_material/", "y", "n", "n", "n", "y", "n", "n", "n", "n", "everything.csv", "sex.csv"]
        
        with mock.patch("builtins.input", mock_input):
            tester = Analyzer()
            
            value_0029 = tester.data.loc[tester.data["utterance"] == 29, "v1_tilt"].item()                      
            value_0059 = tester.data.loc[tester.data["utterance"] == 59, "v1_tilt"].item()  
            value_0079 = tester.data.loc[tester.data["utterance"] == 79, "v1_tilt"].item()  
            
            self.assertAlmostEqual(value_0029, 422.068372224925, places = 1)
            self.assertAlmostEqual(value_0059, 496.260799840674, places = 1)
            self.assertAlmostEqual(value_0079, 525.309714939446, places = 1)
            
    def test_get_center_of_gravity(self):
        """
        Are center of gravity values extracted correctly?
        """
        
        mock_input = Mock()
        mock_input.side_effect = ["test_material/", "y", "n", "n", "n", "n", "y", "n", "n", "n", "everything.csv", "sex.csv"]
        
        with mock.patch("builtins.input", mock_input):
            tester = Analyzer()
            
            value_0048 = tester.data.loc[tester.data["utterance"] == 48, "v1_cog"].item()                      
            value_0053 = tester.data.loc[tester.data["utterance"] == 53, "v1_cog"].item()  
            value_0082 = tester.data.loc[tester.data["utterance"] == 82, "v1_cog"].item()  
            
            self.assertAlmostEqual(value_0048, 202.95697978608, places = 1)
            self.assertAlmostEqual(value_0053, 509.792070942813, places = 1)
            self.assertAlmostEqual(value_0082, 261.330887819259, places = 1)
            
    def test_get_tool_duration(self):
        """
        Are tool durations extracted correctly?
        """
        
        mock_input = Mock()
        mock_input.side_effect = ["test_material/", "y", "n", "n", "n", "n", "n", "y", "n", "n", "everything.csv", "sex.csv"]
        
        with mock.patch("builtins.input", mock_input):
            tester = Analyzer()
            
            value_0025 = tester.data.loc[tester.data["utterance"] == 25, "tool_duration"].item()                      
            value_0037 = tester.data.loc[tester.data["utterance"] == 37, "tool_duration"].item()  
            value_0081 = tester.data.loc[tester.data["utterance"] == 81, "tool_duration"].item()  
            
            self.assertAlmostEqual(value_0025, 292.096, places = 1)
            self.assertAlmostEqual(value_0037, 417.543, places = 1)
            self.assertAlmostEqual(value_0081, 344.82, places = 1)
    
    
    def test_get_target_duration(self):
        """
        Are tool durations extracted correctly?
        """
        
        mock_input = Mock()
        mock_input.side_effect = ["test_material/", "y", "n", "n", "n", "n", "n", "y", "n", "n", "everything.csv", "sex.csv"]
        
        with mock.patch("builtins.input", mock_input):
            tester = Analyzer()
            
            value_0025 = tester.data.loc[tester.data["utterance"] == 25, "target_duration"].item()                      
            value_0037 = tester.data.loc[tester.data["utterance"] == 37, "target_duration"].item()  
            value_0081 = tester.data.loc[tester.data["utterance"] == 81, "target_duration"].item()  
            
            self.assertAlmostEqual(value_0025, 387.95, places = 1)
            self.assertAlmostEqual(value_0037, 348.814, places = 1)
            self.assertAlmostEqual(value_0081, 413.693, places = 1)
            
            
    def test_get_word_duration_ratios(self):
        """
        Are ratios of word durations calculated correctly?
        """
        
        mock_input = Mock()
        mock_input.side_effect = ["test_material/", "y", "n", "n", "n", "n", "n", "y", "n", "n", "everything.csv", "sex.csv"]
        
        with mock.patch("builtins.input", mock_input):
            tester = Analyzer()
            
            value_0025 = tester.data.loc[tester.data["utterance"] == 25, "ratio_word_duration"].item()                      
            value_0037 = tester.data.loc[tester.data["utterance"] == 37, "ratio_word_duration"].item()  
            value_0081 = tester.data.loc[tester.data["utterance"] == 81, "ratio_word_duration"].item()  
            
            self.assertAlmostEqual(value_0025, 0.75282897, places = 1)
            self.assertAlmostEqual(value_0037, 1.1970362, places = 1)
            self.assertAlmostEqual(value_0081, 0.83351664, places = 1)
    

    def test_get_target_height_relative_to_low_end(self):
        """
        Is the target height relative to the low end calculated according to the correct formula?
        12 * log2(pitch_of_target_in_Hz / pitch_of_low_end_in_Hz)
        """
        mock_input = Mock()
        mock_input.side_effect = ["test_material/", "n", "n", "n", "n", "n", "n", "n", "y", "n", "everything.csv", "sex.csv"]
        
        with mock.patch("builtins.input", mock_input):
            tester = Analyzer()
            
            value_0028 = tester.data.loc[tester.data["utterance"] == 28, "exc_target_low_end"].item()                      
            value_0043 = tester.data.loc[tester.data["utterance"] == 43, "exc_target_low_end"].item()  
            value_0094 = tester.data.loc[tester.data["utterance"] == 94, "exc_target_low_end"].item()  
            
            self.assertAlmostEqual(value_0028, -0.88188441, places = 3)
            self.assertAlmostEqual(value_0043, 6.7084764, places = 3)
            self.assertAlmostEqual(value_0094, 3.6141133, places = 3)
        
        
    def test_get_peak_height_relative_to_low_end(self):
        """
        Is the peak height relative to the low end calculated according to the correct formula?
        12 * log2(pitch_of_peak_in_Hz / pitch_of_low_end_in_Hz)
        """
        mock_input = Mock()
        mock_input.side_effect = ["test_material/", "n", "n", "n", "n", "n", "n", "n", "y", "n", "everything.csv", "sex.csv"]
        
        with mock.patch("builtins.input", mock_input):
            tester = Analyzer()
            
            value_0028 = tester.data.loc[tester.data["utterance"] == 28, "exc_peak_low_end"].item()                      
            value_0043 = tester.data.loc[tester.data["utterance"] == 43, "exc_peak_low_end"].item()  
            value_0094 = tester.data.loc[tester.data["utterance"] == 94, "exc_peak_low_end"].item()  
            
            self.assertAlmostEqual(value_0028, -0.88188441, places = 3)
            self.assertAlmostEqual(value_0043, 6.7084764, places = 3)
            self.assertAlmostEqual(value_0094, 6.0587738, places = 3)
            
    def test_get_h1_h2(self):
        """
        Is spectral tilt using H1-H2 calculated correctly?
        """
        mock_input = Mock()
        mock_input.side_effect = ["test_material/", "y", "n", "n", "n", "n", "n", "n", "n", "y", "everything.csv", "sex.csv"]
        
        with mock.patch("builtins.input", mock_input):
            tester = Analyzer()
            
            value_0029 = tester.data.loc[tester.data["utterance"] == 29, "h1_h2"].item()                      
            value_0044 = tester.data.loc[tester.data["utterance"] == 44, "h1_h2"].item()  
            value_0095 = tester.data.loc[tester.data["utterance"] == 95, "h1_h2"].item()
            
            self.assertAlmostEqual(value_0029, 3.783904, places = 3)
            self.assertAlmostEqual(value_0044, 4.6146875, places = 3)
            self.assertAlmostEqual(value_0095, 2.4265053, places = 3)

        