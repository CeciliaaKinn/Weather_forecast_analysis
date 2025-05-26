## Midlertidig. For å teste funksjonene i AnalyzeData. Skal lage en ny klasse med unittest for AnalyzeData senere. 

import sys 
import os 


# Find the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the src folder to sys.path
sys.path.append(os.path.join(project_root, 'src'))

#from services.AnalyzeData import AnalyzeData

# Example of coordinates 
lat = '59.9423'
lon = '10.72'
d_from = '2024-04-01'
d_to = '2024-06-01'

# Data 
#cleaned_df = dp.get_prepared_data()

#data_fetcher = AnalyzeData(lat, lon, d_from, d_to)

# Testing the functions 
#statistics = data_fetcher.statistics()

# Printing the results 
#print(statistics)




