import pandas as pd
from services.DataPreparation import DataPreparation
from services.WindSpeedProcessing import WindSpeedProcessing
from services.LightningProcessing import LightningProcessing

from services.WindSpeedProcessing import WindSpeedProcessing
from services.LightningProcessing import LightningProcessing
from futurePredictions.predictor import WeatherPrediction
from visualization.DataVisualizer import (
    DataVisualizer, 
    Measurements, 
    DataInfo
    )

import matplotlib.pyplot as plt
plt.ion()


def main():
    # Example parameters 
    lat = "60.3833"  # Coordinates: Bergen
    lon = "5.3333"
    d_from_windspeed = "2023-04-01"  # Timeframe
    d_to_windspeed = "2024-05-01"
    d_from_lightning = "2024-04-01"  # Timeframe
    d_to_lightning = "2024-05-01"
    radius = 1  #  Radius for lightningobservations

    own_parameters = str(input("Do you want to choose own parameters? (y/n)"))
    if own_parameters == 'y':
        lat = float(input("Insert latitude as float"))
        lon = float(input("Insert longitude as float"))
        d_from_windspeed = str(input("Insert start date for windspeed (format: yyyy-mm-dd): "))
        d_to_windspeed = str(input("Insert end date for windspeed (format: yyyy-mm-dd): "))
        d_from_lightning = str(input("Insert start date for lightning (format: yyyy-mm-dd): "))
        d_to_lightning = str(input("Insert end date for lightning (format: yyyy-mm-dd): "))
        radius = float(input("Insert radius for lightning observations"))

    wsp = WindSpeedProcessing(lat, lon, d_from_windspeed, d_to_windspeed)
    wsp.save_wind_speed()

    lp = LightningProcessing(lat, lon, d_from_lightning, d_to_lightning, radius)
    lp.save_lightning(['year' ,'month', 'day', 'hour', 'minute', 'second', 'peak current'])



    # Use DataPreparation to load & clean the data
    prep_wind = DataPreparation(
        lat        = lat,
        lon        = lon,
        d_from     = d_from_windspeed,
        d_to       = d_to_windspeed,
        csv_path   = "./data/wind_speed.csv"
    )
    wind_df = prep_wind.get_prepared_data()

    prep_light = DataPreparation(
        lat       = lat,
        lon       = lon,
        d_from    = d_from_lightning,
        d_to      = d_to_lightning,
        json_path = "./data/lightning.json"
    )
    light_df = prep_light.get_prepared_data()

    # Building the two dicts for visualization 
    data_frames = {
        "Wind speed": wind_df,
        "Lightning": light_df,
    }
    measurements = {
        "Wind speed": [Measurements("value", "wind speed (m/s)")],
        "Lightning": [Measurements("peak current", "peak current (kA)")],
    }


    dv = DataVisualizer(data_frames, measurements)
    dv.error_bands_line_plot("Wind speed", y_column="value")
    dv.scatter_plot("Lightning", y_column="peak current")
    dv.correlation_scatter("Wind speed", "value", "Lightning", "peak current", tolerance="72H")


    # Use WeatherPrediction to predict future wind speed and number of lightning strikes
    wp = WeatherPrediction()
    wp.wind_speed_predictor(lat, lon, d_to_windspeed)
    wp.lightning_predictor(lat, lon, d_to_windspeed)



if __name__ == '__main__':
    main() 