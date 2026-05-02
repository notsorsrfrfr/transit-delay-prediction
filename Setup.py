import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime
import requests

def train_and_save():
    print("Loading CTA data...")
    df = pd.read_csv('https://data.cityofchicago.org/api/views/jyb9-n7fm/rows.csv?accessType=DOWNLOAD')
    
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['daytype'].isin(['A','U']).astype(int)
    df['season'] = df['month'].map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3})
    df['route_encoded'] = df['route'].astype('category').cat.codes

    print("Fetching weather data...")
    weather_url = "https://data.cityofchicago.org/api/views/qnmj-8ku6/rows.csv?accessType=DOWNLOAD"
    try:
        weather = pd.read_csv(weather_url)
        weather.columns = weather.columns.str.lower().str.replace(' ', '_')
        weather['date'] = pd.to_datetime(weather['date'])
        weather = weather[['date','temp_high','temp_low','precipitation']].rename(columns={
            'temp_high':'TMAX','temp_low':'TMIN','precipitation':'PRCP'
        })
    except:
        dates = pd.date_range('2001-01-01', '2019-06-28')
        weather = pd.DataFrame({'date': dates, 'TMAX': 15.0, 'TMIN': 5.0, 'PRCP': 0.0})
    
    weather['SNOW'] = 0.0
    weather['AWND'] = 10.0
    weather['date'] = pd.to_datetime(weather['date'])

    merged = df.merge(weather, on='date', how='left')
    for col in ['PRCP','SNOW','AWND','TMAX','TMIN']:
        merged[col] = pd.to_numeric(merged[col], errors='coerce').fillna(0)

    merged['is_rainy'] = (merged['PRCP'] > 0).astype(int)
    merged['is_snowy'] = (merged['SNOW'] > 0).astype(int)

    merged = merged.sort_values(['route','date'])
    merged['rides_lag_1'] = merged.groupby('route')['rides'].shift(1)
    merged['rides_lag_7'] = merged.groupby('route')['rides'].shift(7)
    merged['rides_rolling_7'] = merged.groupby('route')['rides'].transform(
        lambda x: x.shift(1).rolling(7).mean())
    merged = merged.dropna()

    route_avg = merged.groupby('route')['rides'].transform('mean')
    merged['high_load'] = (merged['rides'] > route_avg * 1.2).astype(int)

    features = ['month','day','day_of_week','is_weekend','TMAX','TMIN','PRCP','SNOW','AWND',
                'is_rainy','is_snowy','route_encoded','season','rides_lag_1','rides_lag_7','rides_rolling_7']

    X = merged[features]
    y = merged['high_load']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model...")
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    joblib.dump(model, 'models/best_model.pkl')
    merged.to_csv('data/processed/merged_features.csv', index=False)
    print("Done!")

if __name__ == "__main__":
    train_and_save()