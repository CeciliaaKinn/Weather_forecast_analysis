import pandas as pd
import json
from src.services.FrostClient import FrostClient


class DataProcessingBase:
    def __inint__(self):
        self.client = FrostClient()
        

    def observation_to_df(self, data: list, element: str) -> pd.DataFrame:
        rows = []
        for record in data:
            ref_time = record.get('referenceTime')
            for obs in record.get('observations', []):
                if obs.get('elementId') == element:
                    row = {
                        'referenceTime': ref_time,
                        'value': obs.get('value')
                    }
                    rows.append(row)

        return pd.DataFrame(rows, columns=["referenceTime", "value"])
    

    def observation_to_json(self, data, elements=None):
        if isinstance(elements, str): # Makes sure elements is given as a list
            elements = [elements]
            
        json_data = json.loads(data)
        
        if elements is not None:
            # If elements is specified, we only need the specified data
            filtered_data = []

            # Iterates throug every position in the JSON-data
            for entry in json_data:
                structured_data = {}
                for el in elements:
                    if el in entry:
                        structured_data[el] = entry[el]
                filtered_data.append(structured_data)

            json_data = filtered_data

        return json_data
