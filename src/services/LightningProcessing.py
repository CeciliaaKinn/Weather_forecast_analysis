from services.DataProcessingBase import DataProcessingBase
import json


class LightningProcessing(DataProcessingBase):
    def __init__(self, lat, lon, d_from, d_to, radius):
        super().__init__()
        self.lightning_raw = self.client.getLightning(lat, lon, d_from, d_to, radius)


    def save_lightning(self, element = None):
        json_data = self.observation_to_json(self.lightning_raw, element)
        
        # Lagre JSON-svaret i en fil
        with open('data/lightning.json', 'w') as f:
            json.dump(json_data, f, indent=4)

  