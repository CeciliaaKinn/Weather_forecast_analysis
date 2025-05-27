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


    def add_lightning(self, element = None):
        new_json_data = self.observation_to_json(self.lightning_raw, element)
        
        with open('data/lightning.json', 'r') as f:
            json_data = json.load(f)

        all_json_data = json_data + new_json_data

        # Lagre JSON-svaret i en fil
        with open('data/lightning.json', 'w') as f:
            json.dump(all_json_data, f, indent=4)