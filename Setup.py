import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
from meteostat import Point, Daily
from datetime import datetime

def train_and_save():
    print("Loading CTA data...")
    df = pd.read_csv('https://data.cityofchicago.org/api/views/jyb9-n7fm/rows.csv?accessType=DOWNLOAD')
    
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['daytype'].isin(['A','U']).astype(int)

    print("Fetching weather data...")
    chicago = Point(41.8781, -87.6298)
    weather = Daily(chicago, datetime(2001,1,1), datetime(2019,6,28)).fetch().reset_index()
    weather = weather.rename(columns={'time':'date','tmax':'TMAX','tmin':'TMIN','prcp':'PRCP','snow':'SNOW','wspd':'AWND'})
    weather = weather[['date','TMAX','TMIN','PRCP','SNOW','AWND']]
    weather['date'] = pd.to_datetime(weather['date'])

    merged = df.merge(weather, on='date', how='left')
    for col in ['PRCP','SNOW','AWND','TMAX','TMIN']:
        merged[col] = pd.to_numeric(merged[col], errors='coerce')
        merged[col] = merged[col].fillna(merged[col].mean())

    merged['is_rainy'] = (merged['PRCP'] > 0).astype(int)
    merged['is_snowy'] = (merged['SNOW'] > 0).astype(int)
    merged['route_encoded'] = merged['route'].astype('category').cat.codes
    merged['season'] = merged['month'].map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3})

    merged = merged.sort_values(['route','date'])
    merged['rides_lag_1'] = merged.groupby('route')['rides'].shift(1)
    merged['rides_lag_7'] = merged.groupby('route')['rides'].shift(7)
    merged['rides_rolling_7'] = merged.groupby('route')['rides'].transform(lambda x: x.shift(1).rolling(7).mean())
    merged = merged.dropna()

    route_avg = merged.groupby('route')['rides'].transform('mean')
    merged['high_load'] = (merged['rides'] > route_avg * 1.2).astype(int)

    features = ['month','day','day_of_week','is_weekend','TMAX','TMIN','PRCP','SNOW','AWND',
                'is_rainy','is_snowy','route_encoded','season','rides_lag_1','rides_lag_7','rides_rolling_7']

    X = merged[features]
    y = merged['high_load']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    joblib.dump(model, 'models/best_model.pkl')
    merged.to_csv('data/processed/merged_features.csv', index=False)
    print("Done! Model and data saved.")

if __name__ == "__main__":
    train_and_save()