
"""
Created Sept-2022

@author: 
GROUP 9
Joseph Ndegwa	
Sammy Esese	
Purity Muriithi	
Bravin Daniel	
Nicholas
"""

import streamlit as st
import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
fare = pd.read_csv("train.csv" , nrows = 1000000)
fare_1 = fare.dropna()
pattern_1 = r"[0-9]{4}\S([0-9]{2})"
pattern_2 = r"[0-9]{4}\S[0-9]{2}\S([0-9]{2})"
pattern_3 = r"[0-9]{4}\S[0-9]{2}\S[0-9]{2}.([0-9]{2})"
fare_1["month"] = fare_1["pickup_datetime"].str.extract(pattern_1).astype(int)
fare_1["date"] = fare_1["pickup_datetime"].str.extract(pattern_2).astype(int)
fare_1["time"] = fare_1["pickup_datetime"].str.extract(pattern_3).astype(int)
numerical_col = ['fare_amount', 'pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       'passenger_count', 'month', 'date', 'time']
numerical_fares = fare_1[numerical_col]
def price():
    st.title("TAXI FARE PREDICTIONS NYC")
    months = ["January","February","March","April","May","June","July","August","September","October","November","December"]
    st.selectbox("select month",months)
    p_lo = numerical_fares.iloc[0]["pickup_longitude"]
    p_la = numerical_fares.iloc[0]["pickup_latitude"]
    d_lo = numerical_fares.iloc[0]["dropoff_longitude"]
    d_la = numerical_fares.iloc[0]["dropoff_latitude"]
    p_lon = st.text_input("Enter longitude to start ", p_lo)
    p_lat = st.text_input("Enter latitude to start ", p_la)
    d_lon = st.text_input("Enter longitude coordinate of where the taxi ride will end ", d_lo)
    d_lat = st.text_input("Enter latitude coordinate of where the taxi ride will end", d_la)
    ok = st.button("submit")
    
    knn = KNeighborsRegressor(n_neighbors=10)
    np.random.seed(1)

    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(numerical_fares.index)
    rand_df = numerical_fares.reindex(shuffled_index)

    train_df = rand_df.iloc[0:int(3 / 4 * numerical_fares.shape[0])].copy()  # train dataframe should take 75% of the total totals rows
    test_df = rand_df.iloc[int(3 / 4 * numerical_fares.shape[0]):].copy()  # test df should take the remaining 25% of the rows

    # We Use the fit method to specify the data we want the k-nearest neighbor model to use
    knn.fit(train_df[["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]],
               train_df["fare_amount"])

    # let's Call the predict method to make predictions
    predictions = knn.predict([[p_lon,p_lat,d_lon,d_lat]])
    if ok:
           st.write(f""" ## The predicted fare is $ {predictions[0]} """)
price()
