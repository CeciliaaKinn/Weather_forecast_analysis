
Our data does not contain any missing values, duplicates or outliers. Therefore we made a copy of the csv-file, called 'data_missing_values', where we manually changed the values.  


FOLDER STRUCTURE:

Weather_forecast_analysis/
│
├── data/ # Stored data files
│   ├── data_missing_values.csv
│   ├── lightning.json
│   └── wind_speed.csv
│
├── src/ # Main bulk of code
│   ├── __init__.py
│   │
│   ├── futurePredictions/ # Weather predictions
│   │   ├── __init__.py
│   │   └── predictor.py
│   │
│   ├── services/ # Classes to run the programe
│   │   ├── __init__.py
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
│   │   ├── __init__.py
│   │   └── DataVisualizer.py  
│   │
│   └── main.py 
│
├── tests/ # Unit and integration tests
│   ├── __init__.py
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
