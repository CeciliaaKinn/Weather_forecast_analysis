import pandas as pd 
import requests
from pandasql import sqldf 
import numpy as np
import numpy.ma as ma 
import missingno as msno # for visualizing missing values
import seaborn as sns # Used for visualizations
from dotenv import load_dotenv
import sys 
import os
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split

# Import libraries
import pandas as pd # for dataframes
import numpy as np  # for mathematical operations
import matplotlib.pyplot as plt # for plotting
import missingno as msno # for visualizing missing values

## We have no missing values. We made a copy of the csv-file where we deleted and changed some of the values, and also made duplicates. 

# Find the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the src folder to sys.path
sys.path.append(os.path.join(project_root, 'src')) 

from services.WindSpeedProcessing import WindSpeedProcessing 
from services.FrostClient import FrostClient 
from services.DataProcessingBase import DataProcessingBase

class DataPreparation: 

    """
    This class finds missing values, duplicates etc. and handles the missing values. It prepares the data for visualization 
    and the following analyze. 
    You can choose between a json- or csv-file.

    """

    def __init__(self, lat, lon, d_from, d_to, element='mean(wind_speed P1M)', csv_path = None, json_path = None): 
        load_dotenv()
        self.element = element

        if csv_path: 
            try:
                self.df = pd.read_csv(csv_path)
                self.df['referenceTime'] = pd.to_datetime(self.df['referenceTime'], errors='coerce')
                print(self.df.head())
            except pd.errors.EmptyDataError:
                print("The file is empty or does not include valid values")
        elif json_path: 
            try: 
                self.df = pd.read_json(json_path)

                self.df['referenceTime'] = pd.to_datetime(self.df[['year', 'month', 'day', 'hour', 'minute', 'second']].astype(str).agg('-'.join, axis = 1), format = '%Y-%m-%d-%H-%M-%S') 
                self.df['value'] = pd.to_numeric(self.df['peak current'], errors = 'coerce')
                self.df = self.df[['referenceTime', 'value']]

                self.df = self.df.sort_values('referenceTime').reset_index(drop = True)

                print(self.df.head())
            except: 
                print("The file is empty or does not include valid values")

        else: 
            self.client_id = os.getenv('CLIENTID')
            self.client_credentials = os.getenv('CLIENTCREDENTIALS')
            
            client = FrostClient()
            station_id = client.getClosestWhetherStation(lat, lon)
            self.get_data = client.getWindSpeed(station_id, d_from, d_to)
            
            if self.get_data is not None:
                processor = DataProcessingBase()
                self.df = processor.observation_to_df(self.get_data, self.element)
                self.df['referenceTime'] = pd.to_datetime(self.df['referenceTime'], errors='coerce')

                
            else: 
                self.df = pd.DataFrame() # Empty DataFrame 


    def fetch_data(self, lat, lon, d_from, d_to): 
        """
        Get weather data from GetData and saves in self.data. 
        Parameter elements: elements like temperature and wind speed. 
        Parameter referencetime: time interval for the data. 

        """ 
        client = FrostClient()
        station_id = client.getClosestWhetherStation(lat, lon)
        self.get_data = client.getWindSpeed(station_id, d_from, d_to)

        if self.get_data is not None: 
            print('Data is fetched and saved.')

        else: 
            print('No data was fetched')

        self.df = pd.DataFrame(self.get_data) if self.get_data else None



    def preview_data(self, n = 10):
        """
        Shows the first rows of the data to understand what the data looks like. 
        
        """
        if self.df is not None: 
            return self.df.head(n)
        else: 
            print('No data found')
            return None 
    
        
    def display_monthly_average(self): 
        """
        Shows monthly average values for given elements 

        """ 

        if self.df is not None: 
            print(self.df)

        else: 
            print('No data found')


    def identify_missing_values(self): 
        """
        Identifies missing values. 

        """

        if self.df is not None: 
            return self.df.isnull().sum()
        else: 
            print('No data')
            return None
        
    def find_missing_data(self): 

        """
        Another method for finding missing values.
        Identifying missing columns and values. 
        
        """

        columns_to_check = [col for col in self.df.columns if col != 'referenceTime']

        # Identifying numerical columns

        numerical_columns = [column for column in columns_to_check if np.issubdtype(self.df[column].dtype, np.number)]

        
        # Missing values in numerical columns

        missing_columns = [column for column in numerical_columns if self.df[column].isnull().any()]
        missing_values = {column: self.df[column].isnull().sum() for column in missing_columns}

        # Identifying non-numerical columns
        non_numerical_columns = [column for column in columns_to_check if column not in numerical_columns]

        # Missing values in non-numerical columns (empty strings)
        non_numerical_missing = {column: (self.df[column] == '').sum() for column in non_numerical_columns}

        return missing_values, non_numerical_missing
    
    def mask_missing_values(self): 
        if self.df is None or self.df.empty: 
            print('No data to mask')
            return None 
        masked_data = {}
        for column in self.df.columns: 
            if self.df[column].dtype == object: 
                masked_data[column] = self.df[column] == ''
            else: 
                masked_data[column] = self.df[column].isna()

        return masked_data

    
    def visualize_missing_data(self): 
        """
        Function that visualizes the missing data. 
        1. Matrix of missing data location. 
        2. Bar plot of non-missing values. 
        3. Heatmap. 
        4. Plot of missing values. 
        
        """
        if self.df is not None: 
            # Creates a matrix that shows where misssing values are located. 
            msno.matrix(self.df) 
            plt.show()

            # Creates a bar plot that shows the non-missing values for each column in the dataset. 
            # Identifies the completeness of the data. 
            msno.bar(self.df)
            plt.show()
            
            # Making a heatmap that visualize the correlation of missing values 
            msno.heatmap(self.df)
            plt.show()
            
            # Making a plot to show where the missing values are. 
            self.df["value"].plot()
            plt.xlim(1, 20)
            plt.show()

        else: 
            print('No data to visualize')
    
    
    def find_duplicates(self, subset = 'referenceTime'): 
        """
        This function finds duplicates in the column 'referenceTime', and deletes the second precence. 

        """
        if self.df is not None: 
            duplicates = self.df[self.df.duplicated(subset = subset, keep = 'first')] # Keeps the first value of the duplicates
            print("Duplicates found = ", duplicates)

            df_data_no_duplicates = self.df.drop_duplicates(subset = subset) # Deletes the rest of the duplicates
            return df_data_no_duplicates
        else: 
            print("No data loaded.")
            return None 

    def handle_missing_values(self, strategy = 'drop', column = None, fill_value = None): 
        """
        Function that handles the missing values. 
        Following strategies: 
        1. Drop: reomves the columns with the missing value. 
        2. Fill: changes the missing value with chosen value. 
        3. Forward fill: changes missing value with the value before. 
        4. Backward fill: changes missing value with the value after. 
        5. Interpolate: 
        6. Mean: changes missing value with mean. 
        7. Median: changes missing value with median. 
        """
        if self.df is None: 
            print('No data')
            return None
        
        if strategy == 'drop': # Removes the column with the missing value.
            if column:  
                self.df.dropna(subset = [column])
            else: 
                self.df = self.df.dropna()

        elif strategy == 'fill':  # Changes the missing value with a chosen value. 
            if fill_value is not None: 
                self.df.fillna(fill_value, inplace = True)
            else: 
                raise ValueError("fill_value must be provided when strategy is 'fill'")
        elif strategy == 'forward_fill': # Changes the missing value with the value before. 
            self.df.ffill(inplace = True)
        elif strategy == 'backward_fill': # Changes the missing value with the value after. 
            self.df.bfill(inplace = True)
        elif strategy == 'interpolate': 
            self.df.interpolate(method = 'linear', limit_direction = 'forward', axis = 0) 
            print('Missing values interpolated')
        elif strategy == 'mean': # Changes the missing value with the mean value. 
            if column: 
                mean_value = self.df[column].mean()
                self.df[column] = self.df[column].fillna(mean_value)

            else: 
                raise ValueError('Column must be specified when strategy is mean')
        elif strategy == 'median':  # Changes the missing value with the median value. 
            if column: 
                median_value = self.df[column].median()
                self.df[column] = self.df[column].fillna(median_value)

            else: 
                raise ValueError('Column must be specified when strategy is median')
            
        else: 
            raise ValueError("Choose between strategies: 'drop', 'fill', 'forward_fill', 'backward_fill', 'interpolate' or 'mean'")
        
    
    def find_outliers(self, column = 'value', threshold=3): 
        """
        Function that finds outliers. Checking if the data is between chosen 
        upper and lower limits. The threshold is 3. 

        """


        if self.df is not None and column in self.df.columns: 
            lower_limit = self.df[column].mean() - threshold * self.df[column].std() 
            upper_limit = self.df[column].mean() + threshold * self.df[column].std()
            outliers = self.df[self.df[column].between(lower_limit, upper_limit) == False]

        
        return outliers 


    def find_outliers_iqr(self, column = 'value', threshold=1.5):

        """
        Using the iqr - method to find the outliers. Threshold = 1.5. 

        """
        
        if self.df is not None and column in self.df.columns: 
            q1 = self.df[column].quantile(0.25)
            q3 = self.df[column].quantile(0.75)

            iqr = q3 - q1

            lower_bound = q1 - threshold*iqr
            upper_bound = q3 + threshold*iqr 

            return self.df[(self.df[column] < lower_bound)|self.df[column] > upper_bound]
            
        return None


    def binning_data(self, column = 'value', bins = 5, labels = None): 
        """
        Bins values in given intervals. For example low, medium and high. 
          
        """
        if self.df is not None and column in self.df.columns: 
            if not hasattr(self, 'df') or self.df.empty:
                print("No data")
                return None
            if labels is None: 
                labels = [f"Bin{i+1}" for i in range(bins)]

            self.df['binned'] = pd.cut(self.df[column], bins = bins, labels = labels)
            print(self.df[['value', 'binned']].head())
            return self.df
        else: 
          print('Data is not loaded or column missing.')
          return None 
        
    def execute_sql_query(self, query): 
        """
        Executes a SQL-query on the DataFrame.

        You can use the following SQL-functions:           
          
    1. **SELECT**: Selects data from one or more DataFrames. 
          - Example: "SELECT * FROM df"
          - Selects all data from the DataFrame df. 
    Returns: DataFrame. 
          
        """
        if self.df is not None: 
          return sqldf(query, {'df': self.df})
        else: 
          print('Data is not loaded.')
          return None 
        
    
    def time_series_data_processing(self, datetime_col = 'referenceTime', value_col = 'value', freq = 'D'): 
        if datetime_col not in self.df.columns: 
            print(f"The column {datetime_col} does not exist in the data")
            return None 
        
        self.df[datetime_col] = pd.to_datetime(self.df[datetime_col], errors = 'coerce')
        self.df.dropna(subset = [datetime_col], inplace = True)
        self.df.sort_values(by = datetime_col, inplace = True)
        self.df.set_index(datetime_col, inplace = True)
 
        ts_df = self.df[[value_col]].resample(freq).mean() # Resample

        self.df = ts_df

        return ts_df


    def get_prepared_data(self):
        """
        Returns the prepared data for analyzing and visualization. 
        """
        return self.df

