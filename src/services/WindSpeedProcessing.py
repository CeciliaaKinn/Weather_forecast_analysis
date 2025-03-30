import sys 
import os 

# Find the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the src folder to sys.path
sys.path.append(os.path.join(project_root, 'src')) 

from services.FrostClient import FrostClient
from services.DataProcessingBase import DataProcessingBase


class WindSpeedProcessing(DataProcessingBase):
    def __init__(self, lat, lon, d_from, d_to):
        client = FrostClient()
        station_id = client.getClosestWhetherStation(lat, lon)
        self.wind_speed_raw = client.getWindSpeed(station_id, d_from, d_to)

    def save_wind_speed(self): 
        element = "mean(wind_speed P1D)"
        df = self.observation_to_df(self.wind_speed_raw, element)

        print(df.head())
        df.to_csv("data/wind_speed.csv", index=False)

  