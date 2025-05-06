from dotenv import load_dotenv
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 
import pandas as pd 
import sys 
import os 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score # type: ignore

# Find the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the src folder to sys.path
sys.path.append(os.path.join(project_root, 'src')) 
from services.WindSpeedProcessing import WindSpeedProcessing 
from services.FrostClient import FrostClient 


class AnalyzeData: 

    """
    Class that analyzes the data using mean, median, standard deviation, min, max and modus. 

    Used to understand the trends. 

    """
    
    def __init__(self, df):
        self.df = df.copy()

    
    def fetch_data(self, lat, lon, d_from, d_to): 
        """
        Get weather data from GetData and saves in self.data. 
        Parameter elements: elements like temperature and wind speed. 
        Parameter referencetime: time interval for the data. 

        """ 
        client = FrostClient()
        station_id = client.getClosestWhetherStation(lat, lon)
        self.get_data = client.getWindSpeed(station_id, d_from, d_to)

        if self.get_data is not None: 
            print('Data is fetched and saved.')

        else: 
            print('No data was fetched')
    # Function that finds the mean, median, standard deviation, minimum value and maximum value.
    
    def statistics(self, column = 'value'): 
        self.df[column] = pd.to_numeric(self.df[column], errors = 'coerce')

        stats = {
            'mean': self.df[column].mean(),
            'median': self.df[column].median(), 
            'std': self.df[column].std(), 
            'min': self.df[column].min(), 
            'max': self.df[column].max()
        }
        print(f"Statistics for the column '{column}':\n", stats)
        return stats
    
    # Other methods to find the same statistics. 
    def mean(self): 
        return round(float(np.mean(list(self.df['value'].values()))),2)
    
    def median(self): 
        return round(float(np.median(list(self.df['value'].values()))),2)
    
    def modus(self): 
        return round(float(np.modus))
    

    def correlation_analysis(self, column1 = 'value', column2 = 'temperature'): 
        """
        Finds the correlation between two variables. 
        """

        self.df[column1] = pd.to_numeric(self.df[column1], errors = 'corece')
        self.df[column2] = pd.to_numeric(self.df[column2], errors = 'coerce')

        correlation = self.df[[column1, column2]].corr(method = 'pearson')
        print(f"Correlation between {column1} and {column2}: \n", correlation)
        return correlation 
    
    def transform_data(self, column = 'value', method = 'log'): 
        """
        Funtion that """
        self.df[column] = pd.to_numeric(self.df[column], errors = 'coerce')

        if method == 'log': 
            self.df[column] = np.log1p(self.df[column]) #log(1+x) to handle 0-values. 
        elif method == 'zscore': 
            self.df[column] = (self.df[column] - self.df[column].mean()) / self.df[column].std()

        return self.df
    
    def plot_distribution(self, column):
        """
        Plots the distribution of a variable.

        """
        self.df[column] = pd.to_numeric(self.df[column], errors='coerce')
        sns.histplot(self.df[column].dropna(), kde=True)
        plt.title(f'Fordeling av {column}')
        plt.xlabel(column)
        plt.ylabel('Frekvens')
        plt.grid(True)
        plt.show()
    
    def linear_regression(self, x_col, y_col): 
        """
        Linear regression with the columns from the data. 
        x_col = independent variable (f.ex. temperature). 
        y_col = dependent variable (f.ex. wind speed). 
        
        """
        self.df[x_col] = pd.to_numeric(df[x_col], errors = 'coerce')
        self.df[y_col] = pd.to_numeric(df[y_col], errors = 'coerce')
        self.df.dropna(subset = [x_col, y_col], inplace = True)

        X = self.df[[x_col]].values
        Y = self.df[[y_col]].values 

        model = LinearRegression()
        model.fit(X, Y)
        y_pred = model.predict(X)

        slope = model.coef_[0]
        intercept = model.intercept_
        r2 = r2_score(Y, y_pred)
        mse = mean_squared_error(Y, y_pred)

        #Print 
        print(f"Regression formula: {y_col} = {slope:.2f}*{x_col} + {intercept:.2f}")
        print(f"R^2-value:{r2:.3f}")
        print(f"Mean Squared Error: {mse:.3f}")

        #Plot
        plt.scatter(X, Y, color = 'blue', label = 'Observation')
        plt.plot(X, y_pred, color = 'red', label = 'Regression')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'Regression: {y_col} as a function of {x_col}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return model 
    
    def correlation_matrix(self): 
        """
        Makes a correlatio nmatrix and a heatmap.
        
        """
        numeric_df = self.df.select_dtypes(include = [np.number])
        corr = numeric_df.corr()
        sns.heatmap(corr, annot = True, cmap = 'coolwarm', fmt = '.2f')
        plt.title('Correlation matrix')
        plt.show()

        return corr
    
    

