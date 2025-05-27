import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json 
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys 
import os 

# Find the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the src folder to sys.path
sys.path.append(os.path.join(project_root, 'src'))

from services.WindSpeedProcessing import WindSpeedProcessing
from services.LightningProcessing import LightningProcessing

def collect_training_data(lat, lon, d_to, radius = 0.1):
    # Reference time for training: 1y. Can be changed for more precice predictions.
    date_to = datetime.strptime(d_to, "%Y-%m-%d")
    date_from = date_to - relativedelta(years=1)
    d_from_training = date_from.strftime("%Y-%m-%d")

    training_data_ws = WindSpeedProcessing(lat, lon, d_from_training, d_to)
    training_data_ws.save_wind_speed()
    training_data_l = LightningProcessing(lat, lon, d_from_training, d_to, radius)
    training_data_l.save_lightning()


def wind_speed_predictor(lat, lon, d_to):
    
    # Collect data for training
    collect_training_data(lat, lon, d_to)
    
    df = pd.read_csv('wind_speed.csv') 

    # Giving the colums a more fitting name
    columns = [f"-{29-i}d" for i in range(20)]+['+1d']  

    training_df = pd.DataFrame([
        df['value'].iloc[i:i+31].tolist()
    for i in range(len(df) - 30)
    ])
    training_df.columns = columns 
    
    # Prepares the data for model training
    X = training_df.drop('+1d', axis=1).copy()
    y = training_df['+1d'].copy()   
    model = LinearRegression()
    model.fit(X, y)

    # Split data into training and test sets, allows us to see how precice the model is
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Feature Scaling
    scaler = StandardScaler()
    # Fits the StandardScaler to the data and then transforms the data.
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a linear regression model ## and make predictions on the test set.
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    regr = LinearRegression()
    regr.fit(X_train_scaled_df, y_train)

    # Predict on train set if wanted      
    predict_test_answer = str(input('Do you want to plot the test? y/n'))

    if predict_test_answer == 'y':

        y_pred = regr.predict(X_test_scaled_df)

        # Scatter plot of training data
        plt.figure(figsize=(10, 5))

        # Predictions vs Actual values on test data
        plt.scatter(X_test['-0d'], y_test, color='blue', label='Actual')
        plt.scatter(X_test['-0d'], y_pred, color='red', label='Predicted')

        plt.xlabel('Last Mesurement in m/s')
        plt.ylabel('Wind Speed in m/s')
        plt.title('Actual vs Predicted Wind Speed')
        plt.xlim([0,round(X_test['-0d'].max() + 1)])
        plt.ylim([0,round(max(max(y_test), y_pred.max())) + 1])
        plt.legend()

        plt.show()


    # Reference time for prediction: 30d
    date_to = datetime.strptime(d_to, "%Y-%m-%d")
    date_from = date_to - timedelta(days=30)  
    d_from_prediction = date_from.strftime("%Y-%m-%d")

    # Get reference data 
    reference_data = WindSpeedProcessing(lat, lon, d_from_prediction, d_to)
    reference_data.save_wind_speed()

    # Making a dataframe of wind speed last 30d
    df = pd.read_csv('wind_speed.csv')
    reference_df = pd.DataFrame([df['value']], columns=columns)
    # Scaling to prevent bigger variables from influenceing the predictions more
    reference_df_scaled = pd.DataFrame(scaler.transform(reference_df), columns=reference_df.columns)
    

    # Predict future wind speed
    predicted_ws = pd.DataFrame(columns=[f'+{1+i}d' for i in range(7)])
    
    # Make predictions for the coming 24 hours using the predicted values
    wind_speed_list = np.zeros(7)
    ws_0d = reference_df.loc[0, "-0d"]

    for i in range(7):
        ws = regr.predict(reference_df_scaled).item()
        wind_speed_list[i] = ws
        predicted_ws.loc[len(predicted_ws)] = wind_speed_list
    
        # Moves the values one to the left and ads the predicted value
        reference_df_scaled.iloc[0, :-1] = reference_df_scaled.iloc[0, 1:]
        reference_df_scaled.iloc[0, -1] = ws

    # Adding the current wind speed
    wind_speed_list_complete = np.insert(wind_speed_list, 0, ws_0d)

    # Graf of predicted wind speed next 24h
    plt.figure(figsize=(10,5))

    plt.plot(list(range(8)), wind_speed_list_complete)
    plt.title("Predicted Wind Speed the Next Week")
    plt.xlabel("Time in days")
    plt.ylabel("Wind Speed in m/s")
    plt.xlim([0,7])
    plt.ylim([0, 10])
    plt.grid(True)

    plt.show()


def lightning_predictor(lat, lon, d_to, radius):

    # Collect data for training
    collect_training_data(lat, lon, d_to, radius)


    # Reference data for training, wind speed
    ws_df = pd.read_csv('wind_speed.csv') 
    ws_training_df = pd.DataFrame([
        ws_df['value'].iloc[i:i+10].tolist()
    for i in range(len(ws_df) - 9)
    ])
    # Giving the colums a more fitting name
    ws_training_df.columns = [f"-{9-i}d" for i in range(10)]
        
        
    # Reference data for training, lightning
    with open("./data/lightning.json", "r") as f:
        lightning_data = json.load(f)

    timestamps = [
        datetime(
            int(entry["year"]),
            int(entry["month"]),
            int(entry["day"]),
            int(entry["hour"]),
            int(entry["minute"]),
            int(entry["second"])
        ) for entry in lightning_data
    ]
    ligthning_training_list = []

    date_from = datetime.strptime(ws_df['referenceTime'].iloc[0], "%Y-%m-%dT%H:%M:%S.000Z")
    date_to = datetime.strptime(d_to, "%Y-%m-%dT%H:%M:%S.000Z")

    d_f = date_from
    d_t = date_from + timedelta(days=10)

    for i in range((date_to - date_from).days - 10):
        number_of_ligtning = [date for date in timestamps if d_f <= date <= d_t]
        ligthning_training_list.append(len(number_of_ligtning))
        d_f += timedelta(days=1)
        d_t += timedelta(days=1)

    # Prepares the data for model training
    X = ws_training_df.copy()
    y = ligthning_training_list.copy()   
    model = LinearRegression()
    model.fit(X, y)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Feature Scaling
    scaler = StandardScaler()
    # Fits the StandardScaler to the data and then transforms the data.
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a linear regression model and make predictions on the test set.
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    regr = LinearRegression()
    regr.fit(X_train_scaled_df, y_train)

    # Predict on train set
    y_pred = regr.predict(X_test_scaled_df)

    # Scatter plot of training data
    predict_test_answer = str(input('Do you want to plot the test? y/n'))

    if predict_test_answer == 'y':

        plt.figure(figsize=(10, 5))

        # Predictions vs Actual values on test data
        plt.scatter(X_test['-0d'], y_test, color='blue', label='Actual')
        plt.scatter(X_test['-0d'], y_pred, color='red', label='Predicted')

        plt.xlabel('Last Mesurement in m/s')
        plt.ylabel('Number of Lightning Strikes')
        plt.title('Actual vs Predicted Lightning Strikes')
        plt.xlim([0,round(X_test['-0d'].max() + 1)])
        plt.ylim([-0.5 ,round(max(max(y_test), y_pred.max())) + 1])
        plt.legend()

        plt.show()

    # Reference time for prediction: 10d
    date_to = datetime.strptime(d_to, "%Y-%m-%d")
    date_from = date_to - timedelta(days=10)  
    d_from_prediction = date_from.strftime("%Y-%m-%d")

    # Get reference data 
    reference_data = WindSpeedProcessing(lat, lon, d_from_prediction, d_to)
    reference_data.save_wind_speed()

    # Making a dataframe of wind speed last 30d
    df = pd.read_csv('wind_speed.csv')
    reference_df = pd.DataFrame([df['value']], columns=[f"-{i}d" for i in reversed(range(10))])
    # Scaling to prevent bigger variables from influenceing the predictions more
    reference_df_scaled = pd.DataFrame(scaler.transform(reference_df), columns=reference_df.columns)
    
    # Predicting number of lightning strikes based on wind last 10d
    lightning_nr_scaled_predicted = (regr.predict(reference_df_scaled).item()/10)
    # Devides by 10 because the function predicts number of lightning strikes over a 10d periode
    
    # Graf of wind speed last 10 days and predicted number of lightning strikes the folowing day
    plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

    plt.subplot(gs[0])
    plt.plot([f'-{9 - i}d' for i in range(10)], reference_df.iloc[0])
    plt.title('Wind Speed Last 10 Days')
    plt.xlabel("Day")
    plt.ylabel('Wind Speed in m/s')
    plt.xlim(0,9)
    plt.ylim(0,10)

    plt.subplot(gs[1])
    plt.scatter(0, lightning_nr_scaled_predicted, label = round(lightning_nr_scaled_predicted, 4))
    plt.title("Predicted Number of Lightning\nStrikes the Folowing Day")
    plt.ylabel("Number of Lightning Strikes")
    plt.xlim(-1, 1)
    plt.ylim(0,round(max(max(y_test), y_pred.max())) + 1)
    plt.legend()

    plt.show()