import FrostClient
import DataProcessingBase

class WindSpeedProcessing(DataProcessingBase):
    def __init__(self):
        client = FrostClient()
        id = client.getClosestWhetherStation('59.9423', '10.72')
        self.wind_speed_raw = client.getWindSpeed(id, '2024-04-01', '2024-06-01')

    def save_wind_speed(self):
        elements = None
        fields = None
        df = self.observation_to_df(self.wind_speed_raw, elements, fields)