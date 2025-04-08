from services.WindSpeedProcessing import WindSpeedProcessing
from services.LightningProcessing import LightningProcessing


def main():
    # Example parameters 
    lat = "60.3833"  # Coordinates: Bergen
    lon = "5.3333"
    d_from = "2024-04-01"  # Timeframe
    d_to = "2024-05-01"
    radius = 1  #  Radius for lightningobservations

    own_parameters = str(input("Do you want to choose own parameters? (y/n)"))
    if own_parameters == 'y':
        lat = float(input("Insert latitude as float"))
        lon = float(input("Insert longitude as float"))
        d_from = str(input("Insert start date in yyyy-mm-dd format"))
        d_to = str(input("Insert end date in yyyy-mm-dd format"))
        radius = float(input("Insert radius for lightning observations"))

    wsp = WindSpeedProcessing(lat, lon, d_from, d_to)
    wsp.save_wind_speed()

    lp = LightningProcessing(lat, lon, d_from, d_to, radius)
    lp.save_lightning(['year' ,'month', 'day', 'hour', 'minute', 'second', 'peak current'])

if __name__ == '__main__':
    main() 

