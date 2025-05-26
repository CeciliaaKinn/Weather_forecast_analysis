import sys 
import os 

# Find the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the src folder to sys.path
sys.path.append(os.path.join(project_root, 'src'))

from services.DataPreparation import DataPreparation
from services.AnalyzeData import AnalyzeData

# Paramters 
lat = '60.3833' # Bergen
lon = '5.3333'
d_from = '2023-04-01'
d_to = '2024-05-01'

# Elements to retrieve (monthly mean temperature, wind speed, air pressure and monthly sum precipitation)
# Can use any number and combination of the four
elements = ['mean(air_temperature P1M)', 'mean(wind_speed P1M)', 'sum(precipitation_amount P1M)', 'mean(air_pressure_at_sea_level P1M)']

# Get the data 
csv_path = 'data/data_missing_values.csv'
json_path = 'data/lightning.json'
dp = DataPreparation(lat, lon, d_from, d_to, csv_path = csv_path)

# Testing the functions 
preview_data = dp.preview_data()
missing_data = dp.identify_missing_values()
missing_data2 = dp.find_missing_data()
mask_missing_values = dp.mask_missing_values()
#visualize = dp.visualize_missing_data()
duplicates = dp.find_duplicates()
handle_missing = dp.handle_missing_values()
find_outliers = dp.find_outliers('value')
find_outliers_iqr = dp.find_outliers_iqr()
binning = dp.binning_data(bins = 3)
SQL = dp.execute_sql_query("SELECT * FROM df WHERE value > 5").head()

# Printing the results
print(preview_data)
print(missing_data)
print(missing_data2)
print(duplicates)
print(handle_missing)
print(find_outliers) 
print(find_outliers_iqr)
print(binning)
print("SQL: ", SQL)
#print(visualize)


cleaned_df = dp.get_prepared_data()
analyzer = AnalyzeData(cleaned_df)
analyzer.statistics()
analyzer.plot_distribution('value')
analyzer.linear_regression('referenceTime', 'value') 




