from src.services.DataProcessingBase import DataProcessingBase
import json


class LightningProcessing(DataProcessingBase):
    def __init__(self, d_from, d_to, lat, lon, radius):
        super().__init__()
        self.lightning_raw = self.client.getLightning(d_from, d_to, lon, lat, radius)


    def save_lightning(self, element = None): 
        json_data = self.observation_to_json(self.lightning_raw, element)

        # Lagre JSON-svaret i en fil
        with open('lightning.json', 'w') as f:
            json.dump(json_data, f, indent=4)

  