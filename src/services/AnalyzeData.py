import dotenv 
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 
import pandas as pd 
from FrostClient import FrostClient  
import os 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score # type: ignore

class AnalyzeData: 

    """
    Class that analyzes the data using mean, median, standard deviation, min, max and modus. 

    Used to understand the trends. 

    """
    # Function that finds the mean, median, standard deviation, minimum value and maximum value. 
    def statistics(self, column = 'value'): 
        df_data = pd.DataFrame(self.get_data)
        df_data[column] = pd.to_numeric(df_data[column], error = 'coerce')

        stats = {
            'mean': df_data[column].mean(),
            'median': df_data[column].median(), 
            'std': df_data[column].std(), 
            'min': df_data[column].min(), 
            'max': df_data[column].max()
        }
        print(f"Statistics for the column '{column}':\n", stats)
        return stats
    
    # Other methods to find the same statistics. 
    def mean(self): 
        filepath = 'data/wind_speed.csv'
        self.df = pd.read_csv(filepath)

        return round(float(np.mean(list(self.df['value'].values()))),2)
    
    def median(self): 
        filepath = 'data/wind_speed.csv'
        self.df = pd.read_csv(filepath)

        return round(float(np.median(list(self.df['value'].values()))),2)
    
    def modus(self): 
        filepath = 'data/wind_speed.csv'
        self.df = pd.read_csv(filepath)

        return round(float(np.modus))##
    

    def correlation_analysis(self, column1 = 'value', column2 = 'temperature'): 
        """
        Finds the correlation between two variables. 
        """

        df_data = pd.DataFrame(self.get_data)
        df_data[column1] = pd.to_numeric(df_data[column1], errors = 'corece')
        df_data[column2] = pd.to_numeric(df_data[column2], errors = 'coerce')

        correlation = df_data[[column1, column2]].corr(method = 'pearson')
        print(f"Correlation between {column1} and {column2}: \n", correlation)
        return correlation 
    
    def transform_data(self, column = 'value', method = 'log'): 
        """
        Funtion that """
        df_data = pd.DataFrame(self.get_data)
        df_data[column] = pd.to_numeric(df_data[column], errors = 'coerce')

        if method == 'log': 
            df_data[column] = np.log1p(df_data[column]) #log(1+x) to handle 0-values. 
        elif method == 'zscore': 
            df_data[column] = (df_data[column] - df_data[column].mean()) / df_data[column].std()

        return df_data
    
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

        df = pd.DataFrame(self.get_data)
        df[x_col] = pd.to_numeric(df[x_col], errors = 'coerce')
        df[y_col] = pd.to_numeric(df[y_col], errors = 'coerce')
        df.dropna(subset = [x_col, y_col], inplace = True)

        X = df[[x_col]].values
        Y = df[[y_col]].values 

        model = LinearRegression()
        model.fit(X, Y)
        y_pred = model.predict(X)

        slope = model.coef_[0]
        intercept = model.intercept_
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)

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
        Makes a correlationmatriz and a heatmap.
        
        """

        numeric_df = self.df.select_dtypes(include = [np.number])
        corr = numeric_df.corr()
        sns.heatmap(corr, annot = True, cmap = 'coolwarm', fmt = '.2f')
        plt.title('Correlation matrix')
        plt.show()

        return corr
    
    

