from src.services.WindSpeedProcessing import WindSpeedProcessing


def main():
    wsp = WindSpeedProcessing("59.9423", "10.72", "2024-04-01", "2024-06-01")
    wsp.save_wind_speed()

if __name__ == '__main__':
    main() 

