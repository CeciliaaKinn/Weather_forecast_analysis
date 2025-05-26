import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from dataclasses import dataclass
import matplotlib.dates as mdates
from typing import List, Literal


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
    def __init__(self, data_info: List[DataInfo]):  
        self.data_info = {info.name: info for info in data_info}
        self.data_frames = {
            name: self.prepare_dataframe(self.load_data(info), info)
            for name, info in self.data_info.items()
        }


    def load_data(self, info: DataInfo) -> pd.DataFrame:
        if info.file_type == 'csv':
            return pd.read_csv(info.path)
        if info.file_type == 'json':
            with open(info.path) as f:
                return pd.json_normalize(json.load(f))
        raise ValueError(f'Invalid fileType: {info.file_type}')
    

    def prepare_dataframe(self, df: pd.DataFrame, info: DataInfo) -> pd.DataFrame:
        if {'year','month','day','hour','minute','second'}.issubset(df.columns):
            ts_cols = ['year','month','day','hour','minute','second']
            df['referenceTime'] = pd.to_datetime(df[ts_cols].astype(int))
            df.drop(columns=ts_cols, inplace=True) 
        elif 'referenceTime' not in df.columns:
            raise ValueError(f'Invalid data structure')

        df['referenceTime'] = pd.to_datetime(df['referenceTime'])

        for m in info.measurements:
            if m.name in df.columns:
                df[m.name] = pd.to_numeric(df[m.name], errors='coerce')

        return df.dropna(subset=['referenceTime']).set_index('referenceTime')

    
    def compress_datapoints(self, y_label, df: pd.DataFrame):
        df2 = df.reset_index()  
        df2['month'] = df2['referenceTime'].dt.to_period('M')
        return df2.groupby('month')[y_label].agg(['mean', 'std']).reset_index()

    def error_bands_line_plot(self, data_name, y_label):
        title = next(m.title for m in self.data_info[data_name].measurements if m.name == y_label)

        df = self.data_frames[data_name]
        comp_df = self.compress_datapoints(y_label, df)
        sns.set_theme(style='darkgrid')
        _, ax = plt.subplots(figsize=(12, 5))
        ax.set_title(f"{data_name}")

        ax.plot(
            comp_df['month'].dt.to_timestamp(),
            comp_df['mean'],
            label='Mean'
        )
        
        ax.fill_between(
            comp_df['month'].dt.to_timestamp(),
            comp_df['mean'] - comp_df['std'],
            comp_df['mean'] + comp_df['std'],
            label='Standard deviation',
            alpha=0.2
        )
        ax.set_xlabel('Month')
        ax.set_ylabel(title)
        ax.legend()
        plt.show()


    def scatter_plot(self, data_name: str, y_label: str):
            df = self.data_frames[data_name]
            title = next(m.title for m in self.data_info[data_name].measurements if m.name == y_label)

            _, ax = plt.subplots(figsize=(12, 5))
            sns.set_theme(style="whitegrid", palette="colorblind")

            ax.scatter(df.index, df[y_label], alpha=0.6)
            ax.set_title(f"{data_name}")
            ax.set_xlabel("Time")
            ax.set_ylabel(title)
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.show()
            return ax

     
if __name__ == '__main__': 
    measurements1 = [Measurements('peak current', 'Peak current (kA)')]
    measurements2 = [Measurements('value', 'Wind speed (m/s)')]
    di1 = DataInfo('Lightning', './data/lightning.json', 'json', measurements1)
    di2 = DataInfo('Wind speed', './data/wind_speed.csv', 'csv', measurements2)
    data_info = [di1, di2]

   
    dv = DataVisualizer([di1, di2])
    dv.error_bands_line_plot('Wind speed', y_label='value')
    dv.scatter_plot('Lightning', y_label='peak current')

