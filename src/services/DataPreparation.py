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

## We have no missing values, but we are planning to make a copy of the data 
## and remove data, add duplicates and make outliers. 

# Find the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the src folder to sys.path
sys.path.append(os.path.join(project_root, 'src')) 

from services.WindSpeedProcessing import WindSpeedProcessing 
from services.FrostClient import FrostClient 

class DataPreparation: 

    """
    This class finds missing values, duplicates etc. and handles the missing values"""

    def __init__(self, lat, lon, d_from, d_to):
        #load_dotenv()
        #self.client_id = os.environ['CLIENTID']
        #self.client_credentials = os.environ['CLIENTCREDENTIALS']

        load_dotenv()  # Laster inn miljøvariablene fra .env-filen
        # Sjekk at variablene er lastet riktig
        self.client_id = os.getenv('CLIENTID')
        self.client_credentials = os.getenv('CLIENTCREDENTIALS')

        client = FrostClient()
        station_id = client.getClosestWhetherStation(lat, lon)
        self.wind_speed_raw = client.getWindSpeed(station_id, d_from, d_to)


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


    def preview_data(self, n = 10): 
        df_data = pd.DataFrame(self.get_data)
        if self.get_data is not None: 
        #return self.get_data[:n]
            return df_data.head(n) # Shows the first 10 rows of the data. ## Hvis man har en dataframe. 
        else: 
            print('No data')
            return None 
    
        
    def display_monthly_average(self): 
        """
        Shows monthly average values for given elements 

        """ 

        if self.get_data is not None: 
            print(self.get_data)

        else: 
            print('No data found')

    def save_data_as_csv(self, elements, referencetime): ## Trenger kanskje ikke denne dersom dataen lagres i getdata? Usikker på om man må gjøre det i denne klassen også. 
        """
        Get data and save as csv-file"""

        self.get_data.save_wind_speed(elements, referencetime)
        print('Data saved as csv')


    def identify_missing_values(self):
        """
        Identifies missing values. 
        """

        if self.get_data is not None: 
            df_data = pd.DataFrame(self.get_data)
            missing_values = df_data.isnull().sum()
            return missing_values
        else: 
            print('No data')
            return None
        
    def find_missing_data(self): 

        """Another method for finding missing values.
        Identifying missing columns and values"""

        # Identifying numerical columns

        df_data = pd.DataFrame(self.get_data)

        numerical_columns = [column for column in df_data.columns if np.issubdtype(df_data[column].dtype, np.number)]
        
        # Missing values in numerical columns

        missing_columns = [column for column in numerical_columns if df_data[column].isnull().any()]
        missing_values = {column: df_data[column].isnull().sum() for column in missing_columns}

        # Identifying non-numerical columns
        non_numerical_columns = [column for column in df_data.columns if column not in numerical_columns]

        # Missing values in non-numerical columns (empty strings)
        non_numerical_missing = {column: (df_data[column] == '').sum() for column in non_numerical_columns}

        return missing_values, non_numerical_missing
    
    def mask_missing_values(self): 
        """
        Replacing missing values (NaN or empty strings) with a specific mask. 
        The function also returns a masked version of the data. 

        """

        # Creates masks for all columns. 
        masked_data = {column: self.get_data[column] == '' for column in self.get_data.columns}
        
        # Preview of the first 5 rows in each mask. 
        for name, mask in masked_data.items():
            print(name, mask[:5])

        return masked_data
    
    def visualize_missing_data(self): 
        df_data = pd.DataFrame(self.get_data)

        # Creates a matrix that shows where misssing values are located. 
        print(msno.matrix(df_data)) 
        
        # Creates a bar plot that shows the non-missing values for each column in the dataset. 
        # Identifies the completeness of the data. 
        print(msno.bar(df_data)) 

        # Making a heatmap that visualize the correlation of missing values 
        print(msno.heatmap(df_data))
    
    
    def find_duplicates(self, subset = 'value'): 
        df_data = pd.DataFrame(self.get_data)

        if 'observation' in df_data.columns:
            df_data['value'] = df_data['observations'].apply(lambda x: x[2]['value'] if isinstance(x, list) and len(x) > 2 else None)
            duplicates = df_data[df_data.duplicated(subset=subset)]
            print(duplicates)
            
            df_data_no_duplicates = df_data.drop_duplicates(subset=subset)
            return df_data_no_duplicates
        
        else:
            return None
        #df_data[df_data.duplicated(subset = 'Value')]

        # Total number of rows
        #print("Total number of rows:", df_data.shape[0])
        
        #Find the number of duplicates
        #print("Number of duplicates:", df_data.duplicated(subset = 'Value').sum())
        
        # Find the number of duplicates on the basis of title
       # data_duplicated_title = df_data[df_data.duplicated(subset=["Title"])]
       # print("Number of duplicates on the basis of title:", data_duplicated_title["Value"].sum())
                    
        # Find the number of each duplicated value
        #print("Number of each duplicated value:")
        #print(data_duplicated_title["Value"].value_counts())

        #df_data = df_data.drop_duplicates(subset=["value"], keep='first')
                        
    
    """def handle_duplicates: 
"""
    


  

    """def masked_data(element): 
        data = 

        numerical_columns = [column for column in data.dtype.names if np.issubdtype(data[column].dtype, np.number)]
        missing_columns = [column for column in numerical_columns if np.any(np.isnan(data[column]))]    
        missing_values = {column: np.sum(np.isnan(data[column])) for column in missing_columns}
        non_numerical_columns = [column for column in data.dtype.names if column not in numerical_columns]
        missing_values = {column: np.sum(data[column] == '') for column in non_numerical_columns}

        #return """

    
    # Function that handles missing values. 4 strategies: drop, fill, forward fill or backward fill. 
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
        if strategy == 'drop': # Removes the column with the missing value. 
            self.data.dropna(subset = [column]) if column else self.data.dropna()
        elif strategy == 'fill':  # Change the missing value with a chosen value. 
            if fill_value is not None: 
                self.data = self.data.fillna(fill_value)
            else: 
                raise ValueError("fill_value must be provided when strategy is 'fill'")
        elif strategy == 'forward_fill': # Change the missing value with the value before.  
            self.data = self.data.ffill() 
        elif strategy == 'backward_fill': # Change the missing value with the value after. 
            self.data = self.data.bfill()
        #elif strategy == 'interpolate': 
        #elif strategy == 'mean': 
            #self.data = self.data.mean()
        else: 
            raise ValueError("Choose between strategies: 'drop', 'fill', 'forward_fill', 'backward_fill', 'interpolate', 'mean'")
        
    
    def find_outliers(self, element): 
        """
        Function that finds outliers. Checking if the data is between chosen 
        upper and lower limits. The threshold is 3. 

        """

        df = pd.DataFrame(self.data, columns = [f'{element}'])

        # Calculate lower and upper limits. 

        threshold = 3
        lower_limit = df[f'{element}'].mean() - threshold * df[f'{element}'].std()
        upper_limit = df[f'{element}'].mean() + threshold * df[f'{element}'].std()

        # Outliers
        outliers = df[df[f'{element}'].between(lower_limit, upper_limit) == False]
        
        return outliers 
    
    def find_outliers_iqr(self, threshold = 3): 
        """
        Using the IQR-method to identify the outliers. 
        Parameters: 
         - self
         - threshold: to define the bounds for outliers. Defeault: 3. 

        """ 

        # First step: calculate Q1 (25 %) and Q2 (75 %). 

        data_series = pd.Series(self.data)
        q1 = data_series.quantile(0.25)
        q3 = data_series.quantile(0.75)

        # Calculate IQR. 

        IQR = q3 - q1

        # Bounds for the outliers

        lower_bound = q1 - threshold * IQR
        upper_bound = q3 + threshold * IQR 

        # Identifing the outliers 
        
        outliers = data_series[data_series.between(lower_bound, upper_bound)]

        return outliers 
    
    def binning_data(self): 


    ## Må se mer på denne. Hvordan gjøre slik at den kan håndtere alle tilfellene f.eks.     
    # SQL query. Can select, insert, update, delete or create data. 
    #def execute_sql_query(self, query):
          """Denne funksjonen tar en SQL-spørring som input og utfører den på
          de lokale variablene (DataFrames) ved hjelp av pandasql.
          
          Du kan bruke følgende SQL-funksjoner med denne funksjonen:
          
          
    1. **SELECT**: Henter data fra en eller flere tabeller (DataFrames).
          - Eksempel: "SELECT * FROM df"
          - Henter alle radene fra DataFrame df.
          
    2. **INSERT**: Legger til nye data i en tabell (DataFrame).
          - Merk: `sqldf` støtter ikke direkte `INSERT`, men du kan bruke Pandas-metoder for å legge til data i en DataFrame før du kjører spørringen.
          
          
    3. **UPDATE**: Oppdaterer eksisterende data i en tabell (DataFrame).
         
         - Merk: `sqldf` støtter ikke direkte `UPDATE`, men du kan bruke Pandas-metoder for å oppdatere data etter at du har hentet dem.
       
       
    4. **DELETE**: Sletter data fra en tabell (DataFrame).
       - Merk: `sqldf` har ikke en direkte `DELETE`, men du kan bruke `WHERE` for å filtrere dataene, og deretter oppdatere DataFrame.
       
    5. **CREATE**: Oppretter en ny tabell (DataFrame).
       - Du kan "opprette" en ny DataFrame ved å bruke en SQL-spørring, men den vil være en Pandas DataFrame i minnet, ikke en fysisk tabell i en database.
    
    Argumenter:
    query (str): En SQL-spørring i form av en tekststreng som skal kjøres
    
    Returnerer:
    DataFrame: Resultatet av SQL-spørringen som en Pandas DataFrame"""
          
         # return sqldf(query, locals()) ## Må se litt mer på hvordan vi bruker SQL.


## Time series data processing 
## signal processing - ikke nødvendigvis relevant 