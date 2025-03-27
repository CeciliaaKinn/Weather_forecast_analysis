import requests
from requests.auth import HTTPBasicAuth
import os
from dotenv import load_dotenv

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

if __name__ == '__main__':
    client = FrostClient()
    id = client.getClosestWhetherStation('59.9423', '10.72')
    print(client.getWindSpeed(id, '2024-04-01', '2024-06-01')) # SN18700