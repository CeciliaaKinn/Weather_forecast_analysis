from services.FrostClient import FrostClient
from services.DataProcessingBase import DataProcessingBase


class WindSpeedProcessing(DataProcessingBase):
    def __init__(self, lat, lon, d_from, d_to):
        super().__init__()
        station_id = self.client.getClosestWhetherStation(lat, lon)
        self.wind_speed_raw = self.client.getWindSpeed(station_id, d_from, d_to)

    def save_wind_speed(self): 
        element = "mean(wind_speed P1D)"
        df = self.observation_to_df(self.wind_speed_raw, element)

        print(df.head())
        df.to_csv("data/wind_speed.csv", index=False)

  