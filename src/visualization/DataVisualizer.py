import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from dataclasses import dataclass
import matplotlib.dates as mdates
from typing import Dict, List, Literal, Union



@dataclass
class Measurements:
    name: str
    title: str

@dataclass
class DataInfo:
    name: str
    path: str
    file_type: Literal['csv', 'json'] 
    measurements: list[Measurements]


class DataVisualizer:

    def __init__(self,
                 data_frames: dict[str, pd.DataFrame],
                 measurements: dict[str, list[Measurements]]):
        self.data_frames = data_frames
        self.data_info = measurements


    def get_title(self, data_name: str, col_name: str) -> str:
        """ 
        Returns the user friendly column name for the specified data collection (data_name)
        """
        return next(m.title for m in self.data_info[data_name] if m.name == col_name)

    
    def compress_datapoints(self, y_label, df: pd.DataFrame):
        """
        Find standard derivation and mean for data measurements per month
        """
        df2 = df.reset_index()  
        df2['month'] = df2['referenceTime'].dt.to_period('M')
        return df2.groupby('month')[y_label].agg(['mean', 'std']).reset_index()
             

    def error_bands_line_plot(self, data_name, y_column):
        """
        Plot standard derivation and mean of the y-values per month 
        """
        title = self.get_title(data_name, y_column)

        df = self.data_frames[data_name]
        compressed_df = self.compress_datapoints(y_column, df)
        sns.set_theme(style='darkgrid')
        _, ax = plt.subplots(figsize=(12, 5))
        ax.set_title(f"{data_name}")

        ax.plot(
            compressed_df['month'].dt.to_timestamp(),
            compressed_df['mean'],
            label='Mean'
        )
        
        ax.fill_between(
            compressed_df['month'].dt.to_timestamp(),
            compressed_df['mean'] - compressed_df['std'],
            compressed_df['mean'] + compressed_df['std'],
            label='Standard deviation',
            alpha=0.2
        )
        ax.set_xlabel('month')
        ax.set_ylabel(title)
        ax.legend()
        plt.show()
        return ax


    def scatter_plot(self, data_name: str, y_column: str):
        """
        Draw a scatter plot of y-values over time for the specified data collection (data_name)
        """
        df = self.data_frames[data_name]
        title = next(m.title for m in self.data_info[data_name] if m.name == y_column)

        _, ax = plt.subplots(figsize=(12, 5))
        sns.set_theme(style="darkgrid", palette="colorblind")

        ax.scatter(df.index, df[y_column], alpha=0.6)
        ax.set_title(f"{data_name}")
        ax.set_xlabel("time")
        ax.set_ylabel(title)
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.show()
        return ax



    def correlation_scatter(self,
                            x_data_name: str,
                            x_column: str,
                            y_data_name: str,
                            y_column: str,
                            tolerance: Union[str, pd.Timedelta] = "5min"):
        """
        Show the correlation between the y and x data points 
        It includes the uncertainty in the estimate of the regression
        """
        df_x = (self.data_frames[x_data_name][[x_column]]
                .rename(columns={x_column: "x"})
                .sort_index()
                .reset_index())  

        df_y = (self.data_frames[y_data_name][[y_column]]
                .rename(columns={y_column: "y"})
                .sort_index()
                .reset_index())
  
        df = pd.merge_asof(df_x, df_y,
                        on="referenceTime",
                        direction="nearest",
                        tolerance=pd.Timedelta(tolerance)) \
            .dropna()
        if df.empty:
            print("No matches within the selected tolerance, try a larger window.")
            return

        r = df[["x", "y"]].corr().loc["x", "y"]

        sns.set_theme(style="darkgrid", palette="colorblind")
        _, ax = plt.subplots(figsize=(12, 5))
        sns.regplot(data=df, x="x", y="y", ax=ax)

        x_title = self.get_title(x_data_name, x_column)
        y_title = self.get_title(y_data_name, y_column)
        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)
        ax.set_title(
            f"Relationship between {x_title} and {y_title} (r = {r:.2f})"
        )
        plt.tight_layout()
        plt.show()
        return ax

