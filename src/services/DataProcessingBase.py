import pandas as pd

#class DataProcessingBase:
#    def observation_to_df(self, json, element: str, fields: list[str]) -> pd.DataFrame:
#        # TODO for loop with over the json file 
#       pass




class DataProcessingBase: 
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
    
