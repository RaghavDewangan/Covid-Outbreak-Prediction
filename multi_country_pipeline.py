import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

def load_owid_data(): # load the OWID dataset
    url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    df = pd.read_csv(url, parse_dates=['date'])
    return df

def prepare_country_data(df, country, scaler): # transformation pipeline (Scale data with set scaler)
    features = ['new_cases', 'stringency_index', 'people_vaccinated_per_hundred']
    df_c = df[df['location'] == country][['date'] + features].copy() # take the proper country, and take Dates and other features , need dates as indices
    df_c.set_index('date', inplace=True)

    df_c.fillna(method='ffill', inplace=True)
    df_c.fillna(method='bfill', inplace=True) # fill the nan values front and back
    
    df_c = df_c[df_c['new_cases'] > 0] # pull significant values for model to be trained upon

    scaled = scaler.transform(df_c) # scale the df with specified scaler
    df_scaled = pd.DataFrame(scaled, columns=features, index=df_c.index)
    df_scaled['country'] = country

    return df_scaled

def combine_countries(df, countries, scaler):
    all_data = [] # for concatenated holistic data
    for c in countries:
        df_c = prepare_country_data(df, c, scaler)
        all_data.append(df_c)
    combined = pd.concat(all_data)
    return combined.reset_index()

def encode_country(df, all_countries=None):
    dummies = pd.get_dummies(df['country'], prefix='country')
    if all_countries:
        for c in all_countries:
            col = f'country_{c}'
            if col not in dummies:
                dummies[col] = 0
    df = pd.concat([df.drop(columns=['country']), dummies], axis=1)
    return df  # encode all countries (for streamlit to work)

def create_sliding_windows(df, window_size, target_col='new_cases'):
    X, y = [], []
    df = df.copy().reset_index(drop=True)
    feature_cols = df.columns.drop(['date']) if 'date' in df.columns else df.columns.drop(target_col)

    for i in range(len(df) - window_size):
        window = df.iloc[i:i+window_size][feature_cols].values
        target = df.iloc[i+window_size][target_col]
        X.append(window)
        y.append(target)
        
    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.float32) # so tensorflow has no issues with "object" type when taking x_Train y_train etc...

    return np.array(X), np.array(y)

#### LTSM Model Builder

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

#### scikit learn compatible custome transformer, can be plugged into sklearn pipelines
class MultiCountryCOVIDPreprocessor(BaseEstimator, TransformerMixin): 
    def __init__(self, countries, window_size=14):
        self.countries = countries
        self.window_size = window_size
        self.scaler = None

    def fit(self, X, y=None):
        features = ['new_cases', 'stringency_index', 'people_vaccinated_per_hundred'] # load and scale covid 19 data for multiple countries
        raw_concat = pd.concat([
            X[X['location'] == c][features] for c in self.countries
        ])
        self.scaler = MinMaxScaler().fit(raw_concat)
        return self 

    def transform(self, X): # one hot encodes country col
        df_combined = combine_countries(X, self.countries, self.scaler)
        df_encoded = encode_country(df_combined)
        X_windowed, y = create_sliding_windows(df_encoded, self.window_size) # creates sliding windows, returns np arrays for modelling
        return X_windowed, y