# UNITTEST FOR DATAPREPARATION
import sys 
import os
import pandas as pd 
import unittest
from datetime import datetime

# Find the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the src folder to sys.path
sys.path.append(os.path.join(project_root, 'src')) 

from services.DataPreparation import DataPreparation 


class DataPreparation_tester: 
    def setUp(self):
        self.data_prep = DataPreparation(lat=60.0, lon=10.0, d_from="2025-01-01", d_to="2025-01-31")

    
    def test_preview_data(self): 
        preview = self.data_prep.preview_data(5)
        self.assertIsInstance(preview, pd.DataFrame)
        self.assertLessEqual(len(preview), 5)
    
    def test_identify_missing_values(self): 
        missing = self.data_prep.identify_missing_values()
        self.assertIsInstance(missing, pd.Series)
        self.assertEqual(missing.sum(), 0) # Checking that there is no missing values (the dictionary is empty). 

    def test_find_missing_data(self): 
        missing_values, non_numerical_missing = self.data_prep.find_missing_data()
        self.assertIsInstance(missing_values, dict)
        self.assertIsInstance(non_numerical_missing, dict)
        self.assertEqual(len(missing_values), 0) # Checking that there is no missing values. 
        self.assertEqual(len(non_numerical_missing), 0) # Checking that there is no missing numerical values. 

    def test_mask_missing_values(self): 
        masked_data = self.data_prep.mask_missing_values()
        self.assertIsInstance(masked_data, dict)
        self.assertTrue(all(isinstance(v, pd.Series) for v in masked_data.values(())))

    def test_visualize_missing_data(self): 
        self.data_prep.visualize_missing_data()


        #### Må se mer på denne 

    def test_find_duplicates(self): 
        self.data_prep.df = self.data_prep.df.append(self.data_prep.df.isloc[0], ignore_index = True) # Gets the first row and adds a duplicate of the first row. 
        df_no_duplicates = self.data_prep.find_duplicates() 
        self.assertIsInstance(df_no_duplicates, pd.DataFrame)
        self.assertEqual(df_no_duplicates.shape[0], 2) 

    def test_handle_missing_values_drop(self): 
        self.data_prep.df.loc[0, 'value'] = None
        self.data_prep.handle_missing_values(strategy = 'drop')
        self.assertEqual(self.data_prep.df.shape[0], 1)

    def test_handle_missing_values_fill(self): 
        self.data_prep.df.loc[0, 'value'] = None
        self.data_prep.handle_missing_values(strategy = 'fill', fill_value = 0)
        self.assertEqual(self.data_prep.df.loc[0, 'value'], 0)

    def test_find_outliers(self): 
        outliers = self.data_prep.find_outliers()
        self.assertIsInstance(outliers, pd.DataFrame)
        self.assertEqual(outliers.shape[0], 0)

    def test_binning_data(self): 
        self.data_prep.binning_data(bins = 3)
        self.assertIn('binned', self.data_prep.df.columns)
    
    def test_execute_sql_query(self): 
        query = "SELECT * FROM df WHERE value > 5"
        result = self.data_prep.execute_sql_query(query)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[0], 1)

    def test_time_series_data_processing(self):
        ts_df = self.data_prep.time_series_data_processing()
        self.assertIsInstance(ts_df, pd.DataFrame)
        self.assertEqual(ts_df.shape[0], 1)

    def test_get_prepared_data(self):
        prepared_data = self.data_prep.get_prepared_data()
        self.assertIsInstance(prepared_data, pd.DataFrame)
        self.assertEqual(prepared_data.shape[0], 2) # Må endre på dennne. 

    if __name__ == '__main__': 
        unittest.main()



