from services.WindSpeedProcessing import WindSpeedProcessing
from services.LightningProcessing import LightningProcessing


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

if __name__ == '__main__':
    main() 

