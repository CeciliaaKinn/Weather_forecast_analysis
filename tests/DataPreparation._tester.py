import sys 
import os 

# Find the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the src folder to sys.path
sys.path.append(os.path.join(project_root, 'src'))

from services.DataPreparation import DataPreparation

# Instantiate the class

lat = '59.9423'
lon = '10.72'
d_from = '2024-04-01'
d_to = '2024-06-01'
data_fetcher = DataPreparation(lat, lon, d_from, d_to)

# Elements to retrieve (monthly mean temperature, wind speed, air pressure and monthly sum precipitation)
# Can use any number and combination of the four
elements = ['mean(air_temperature P1M)', 'mean(wind_speed P1M)', 'sum(precipitation_amount P1M)', 'mean(air_pressure_at_sea_level P1M)']

# Date range
referencetime = '2010-01-01/2010-12-31'  # Example

# Get the data in JSON format
data = data_fetcher.fetch_data(lat, lon, d_from, d_to)

# Display the monthly averages or sums for the requested elements
#data_fetcher.display_monthly_average(lat, lon, d_from, d_to)

#dp = DataPreparation(api_url = 'https://frost.met.no/observations/v0.jsonld')
#dp.data = json_data 


preview_data = data_fetcher.preview_data(data)
missing_data = data_fetcher.identify_missing_values()
missing_data2 = data_fetcher.find_missing_data()

# Print the results
print(preview_data)
print(missing_data)
print(missing_data2)
