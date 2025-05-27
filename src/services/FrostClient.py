import requests
from requests.auth import HTTPBasicAuth
import os
from dotenv import load_dotenv
import json
from shapely.wkt import loads

class FrostClient:
    def __init__(self):
        load_dotenv()
        self.client_id = os.environ['CLIENTID']
        self.client_credentials = os.environ['CLIENTCREDENTIALS']


    def getClosestWhetherStation(self, lat, lon):
        params = {
            'geometry': f'nearest(POINT({lon} {lat}))'
        }
        r = requests.get(
            'https://frost.met.no/sources/v0.jsonld',
            params=params,
            auth=HTTPBasicAuth(self.client_id, self.client_credentials)
        )
        r.raise_for_status() 
        r = r.json()
        if not r.get('data'):
            raise Exception('No source found for the specified location.')
        return r['data'][0]['id']


    def getWindSpeed(self, ws_id, d_from, d_to):
        params = {
            'sources': ws_id,
            'elements': 'mean(wind_speed P1D)',
            'referencetime': f'{d_from}/{d_to}'
        }

        r = requests.get(
            'https://frost.met.no/observations/v0.jsonld',
            params=params,
            auth=HTTPBasicAuth(self.client_id, self.client_credentials)
        )
        r.raise_for_status() 
        r = r.json()
        if not r.get('data'):
            raise Exception('No source found for the specified location.')
        return r['data']
    

    def getAirTemperature(self, ws_id, d_from, d_to):
            # To be implemented
            pass
    

    def getPolygon(self, lat, lon, radius):
        # Decides the area for observtions
        lon_min = str(round(float(lon) - radius, 6))
        lon_max = str(round(float(lon) + radius, 6))
        lat_min = str(round(float(lat) - radius, 6))
        lat_max = str(round(float(lat) + radius, 6))
        wkt_polygon = f'''POLYGON(({lon_min} {lat_min}, {lon_min} {lat_max}, {lon_max} {lat_max}, {lon_max} {lat_min}, {lon_min} {lat_min}))'''
        polygon = loads(wkt_polygon)
        return polygon
    

    def getLightning(self, lat, lon, d_from, d_to, radius = 1):
        params = {
            'referencetime': f'{d_from}/{d_to}',
            'maxage': '',
            'geometry': self.getPolygon(lat, lon, radius),
        }
        
        r = requests.get(
            'https://frost.met.no/lightning/v0.jsonld',
            params=params,
            auth=(self.client_id, self.client_credentials)
        )

        if r.status_code == 200:
            data = r.text  # Converts the data to a more handelable format
            rows = data.strip().split('\n')  # Splits the data into rown (by lightning)
            
            headers = [
                'version', 'year', 'month', 'day', 'hour', 'minute', 'second', 'nanoseconds', 
                'latitude', 'longitude', 'peak current', 'degrees of freedom', 'semi-minor axis',
                'semi-major axis', 'ellipse angle', 'multiplicity', 'number of sensors', 
                'chi-square value', 'rise time', 'peak-to-zero time', 'timing indicator', 
                'signal indicator', 'angle indicator', 'cloud indicator', 'max rate-of-rise'
            ]

            data_list = []

            for row in rows:
                values = row.split()  # Split the row by spaces to get individual values
                
                # Create a dictionary mapping header names to the values
                data_dict = dict(zip(headers, values))
                    
                # Add the entire data_dict to the data_list (no modification)
                data_list.append(data_dict)

            # Convert the list of dictionaries to JSON
            json_data = json.dumps(data_list, indent=4)
            return json_data
        else:
            print(f"Error! Returned status code {r.status_code}")
            print(f"Message: {r.json()['error']['message']}")
            print(f"Reason: {r.json()['error']['reason']}")
            return None


if __name__ == '__main__':
    client = FrostClient()
    id = client.getClosestWhetherStation('60.3833', '5.3333')
    print(client.getWindSpeed(id, '2024-04-01', '2024-05-01')) # SN18700
