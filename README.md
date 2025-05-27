WEATHER FORECAST ANALYSIS

DESCRIPTION: 
This program is designed to fetch real-time environmental data (lightning strikes and wind speed), perform statistical analysis, visualize trends, and make future predictions using the data. 

FEATURES: 
- Fetch live lightning strikes and wind speed data from external APIs
- Perform statistical summaries and trend analysis (mean, median, standard derivation, trends)
- Visualize data 
- Predict future lightning and wind speed 
- Data storage in JSON and CSV formats for use 

STATISTICS: 
- Mean, median, standard derivation, min, max and modus 
- Correlation analysis 
- Correlation matrix
- Linear regression 

VIZUALISATION: 
- Shatter plot 
- Correlation plot (between lightning peak current and wind speed)
- Error lines band plot 

PREDICTIONS: 
- Wind speed for the following week 
- Number of lightning strikes the following day 

TECHNOLOGIES USED: 
- Python 
- Pandas and Numpy 
- Matplotlib and Seaborn 
- Scikit-learn (for predictions)
- Requests (for API-calls)
- Jupyter Notebooks (for developement and testing)

EXAMPLE:
JSON-format: {
        "year": "2024",
        "month": "04",
        "day": "10",
        "hour": "08",
        "minute": "07",
        "second": "52",
        "peak current": "-6"
    }

FAQ
Q: Can I use this with different APIs?
A: Yes. Modify FrostClient. 
Q: How often is the data fetched?
A: The data is fetched every hour. 

NOTE: 
Our data does not contain any missing values, duplicates or outliers. Therefore we made a copy of the csv-file, called 'data_missing_values', where we manually changed the values. This file contains missing 


FOLDER STRUCTURE:

Weather_forecast_analysis/
│
├── data/ # Stored data files
│   ├── data_missing_values.csv
│   ├── lightning.json
│   └── wind_speed.csv
│
├── src/ # Main bulk of code
│   ├── _init_.py
│   │
│   ├── futurePredictions/ # Weather predictions
│   │   ├── _init_.py
│   │   └── predictor.py
│   │
│   ├── services/ # Classes to run the programe
│   │   ├── _init_.py
│   │   ├── AnalyzeData.py # Statistics
│   │   ├── DataPreparation.py # Preparing the data for analyzing and visualizing
│   │   ├── DataProcessingBase.py # the shared logic between the processing classes
│   │   ├── FrostClient.py # Data source
│   │   ├── LightningProcessing.py
│   │   └── WindSpeedProcessing.py 
│   │   # For the ...Processing.py-files: Use Frost client to get data from API and 
│   │     use file_manager to sava data
│   │ 
│   ├── visualization/ # Presents the data in a clear and structured way
│   │   ├── _init_.py
│   │   └── DataVisualizer.py  
│   │
│   └── main.py 
│
├── tests/ # Unit and integration tests
│   ├── _init_.py
│   ├── unittest
│   │   ├── AnalyzeData_test.py
│   │   ├── DataPreparation_test.py
│   │   └── DataVisualizer_test.py
│   ├── AnalyzeData_tester.py
│   ├── Tester_analyze_preparation.py
│   ├── WindSpeedProcessing_tester.py 

│
├── .env # In .gitignore. Here we store password and clientID
├── .gitignore # Ignores client id and client credentials from .env. It also contains everything from the data folder.
├── requirements.txt # Project dependencies (packages)
└── README.md # Project overview
init__.py
