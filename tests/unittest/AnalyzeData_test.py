 # UNITTEST FOR ANALYZEDATA
import sys 
import os
import pandas as pd 
import unittest 
import numpy as np 

# Find the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the src folder to sys.path
sys.path.append(os.path.join(project_root, 'src')) 

from services.DataPreparation import DataPreparation
from services.AnalyzeData import AnalyzeData  


class DataPreparation_tester(unittest.TestCase): 
    def setUp(self):
        self.data_prep = DataPreparation(lat=60.0, lon=10.0, d_from="2025-01-01", d_to="2025-01-31")

    def test_statistics(self):
        # Using assertAlmostEqual since we have float 
        stats = self.analyzer.statistics('value')
        self.assertAlmostEqual(stats['mean'], 3.0)
        self.assertAlmostEqual(stats['median'], 3.0)
        self.assertAlmostEqual(stats['std'], np.std([1, 2, 3, 4, 5], ddof=1))
        self.assertEqual(stats['min'], 1)
        self.assertEqual(stats['max'], 5)

    def test_statistics_with_non_numeric_column(self): 
        df = self.data_prep.copy()
        df['value'] = ['a', 'b', 'c', 'd', 'e']
        analyze = AnalyzeData(df)
        stats = analyze.statistics('value')
        # Statistics should be NaN for mean, median, std, min, max
        for key in ['mean', 'median', 'std', 'min', 'max']: 
            self.assertTrue(np.isnan(stats[key]), f"{key} should be NaN") 

    def test_correlation_analysis_with_non_numeric(self): 
        df = self.data_prep.copy()
        df['value'] = ['a', 'b', 'c', 'd', 'e']
        analyze = AnalyzeData(df)
        corr = analyze.correlation_analysis('value', 'temperature')
        # The correlation should be NaN since 'value' is non-numeric 
        self.assertTrue(corr.isnull().values.any())

    def test_linear_regression_with_missing_columns(self): 
        # Testing that linear_regression handles missing columns 
        df = pd.DataFrame({'value': [1, 2, 3]}) # Missing 'temperature'
        analyze = AnalyzeData(df)
        with self.assertRaises(KeyError): 
            analyze.linear_regression('value', 'temperature')

    def test_transform_regression_with_nonexistent_columns(self): 
        # Trying to plot a column that does not exist 
        with self.assertRaises(KeyError): 
            self.analyze.plot_distribution('nonexistent_column')

    def test_correlation_analysis(self): 
        corr = self.analyzer.correlation_analysis('value', 'temperature')
        corr_value = corr.loc['value', 'temperature']
        self.assertAlmostEqual(corr_value, -1.0, places=2) 

    def test_transform_data_log(self):
        transformed_df = self.analyzer.transform_data('value', method='log')
        expected = np.log1p([1, 2, 3, 4, 5])
        np.testing.assert_array_almost_equal(transformed_df['value'].values, expected)

    def test_transform_data_zscore(self):
        transformed_df = self.analyzer.transform_data('value', method='zscore')
        expected = (np.array([1, 2, 3, 4, 5]) - 3) / np.std([1, 2, 3, 4, 5], ddof=1)
        np.testing.assert_array_almost_equal(transformed_df['value'].values, expected)

    def test_linear_regression(self):
        model = self.analyzer.linear_regression('temperature', 'value')
        self.assertIsInstance(model, LinearRegression)
        self.assertAlmostEqual(float(model.coef_[0]), -1.0, places=2)
        self.assertAlmostEqual(float(model.intercept_), 6.0, places=2)

if __name__ == '__main__':
    unittest.main()
