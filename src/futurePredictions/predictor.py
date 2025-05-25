import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import sys 
import os 

# Find the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the src folder to sys.path
sys.path.append(os.path.join(project_root, 'src'))

from services.WindSpeedProcessing import WindSpeedProcessing
from services.AirTemperatureProcessing import AirTemperatureProcessing
from services.LightningProcessing import LightningProcessing


def collect_data(what):
    pass


def wind_speed_predictor(lat, lon, d_to):
    # Reference time for training: 1y
    if int(d_to[:4])%4 == 0 and d_to[:4] != '2000' and d_to[4:] == '-02-29':  
        d_from = f'{int(d_to[:4] - 1)}-02-28'
    else:    
        d_from = f'{int(d_to[:4]) - 1}{d_to[4:]}'  # 1 year reference time
    training_data = WindSpeedProcessing(lat, lon, d_from, d_to)
    training_data.save_wind_speed()
    
    df = pd.read_csv('wind_speed.csv')
   
    reference_df = pd.DataFrame([
        df['value'].iloc[i:i+24].tolist()
    for i in range(len(df) - 23)
    ])

    # Giving the colums a more fitting name
    reference_df.columns = [f"-{23-i}t" for i in range(24)]

    refrence_columns = []
    for i in range(23):
        refrence_columns.append(f'-{23-i}t')

    
    # Prepares the data for model training
    X = reference_df[refrence_columns].copy()  # Feature
    y = reference_df['-0t'].copy()      # Target (wind speed)
    model = LinearRegression()
    model.fit(X, y)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Reference time for prediction: 24h
    if d_to[-2:] == '01':
        if d_to[5:7] == '01':
            d_from = f'{int(d_to[:4])-1}-12-30'
        elif d_to[5:7] == '05' or '07' or '09' or '11':
            d_from = f'{d_to[:5]}0{int(d_to[5:7]) - 1}-30'
        elif d_to[5:7] == '03':
            if int(d_to[:4])%4 == 0 and (d_to[:4]) != '2000':
                d_from = f'{d_to[:5]}0{int(d_to[5:7]) - 1}-29'
            else:
                d_from = f'{d_to[:5]}0{int(d_to[5:7]) - 1}-28'
        else:
            d_from = f'{d_to[:5]}0{int(d_to[5:7]) - 1}-31'
    else:    
        d_from = f'{d_to[:-1]}{int(d_to[-1:]) - 1}'
    training_data = WindSpeedProcessing(lat, lon, d_from, d_to)
    training_data.save_wind_speed()




def air_temperature_predictor(lat, lon, d_to):
    pass 

    d_from = f'{int(d_to[:4]) - 10}{d_to[4:]}'
    atp = AirTemperatureProcessing(lat, lon, d_from, d_to)
    atp.save_air_temperature 
    
    df = pd.read_csv('wind_speed.csv', parse_dates=['referenceTime'])
   
    for i in df:
        df_2 = 2
    
    # Prepares the data for model training
    X = df[['days_since_start']]  # Feature
    y = df['value']               # Target (wind speed)
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions for existing data
    df['predicted'] = model.predict(X)

    # Plot original vs. prediction
    plt.scatter(df['referenceTime'], df['value'], label='Observasjoner', s=10)
    plt.plot(df['referenceTime'], df['predicted'], color='red', label='Modell')
    plt.xlabel('Dato')
    plt.ylabel('Vindstyrke (m/s)')
    plt.title('Wind Speed – Observasjoner og Prediksjon')
    plt.legend()
    plt.show()


def lightning_predictor():
    pass