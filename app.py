import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Metrics Correlations and Decay Effect")

uploaded_file = st.file_uploader("Upload a CSV file:", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'], dayfirst=True)
    df.set_index('Date', inplace=True)

    def apply_decay(data, decay_rate):
        decayed_data = data.copy()
        for i in range(1, len(data)):
            decayed_data.iloc[i] = decayed_data.iloc[i] + decayed_data.iloc[i - 1] * decay_rate
        return decayed_data

    def find_time_to_reach_90_percent(decay_rate):
        time_to_reach_90_percent = int(np.ceil(np.log(0.1) / np.log(decay_rate)))
        return time_to_reach_90_percent

    decay_rate = 0.9

    for col in df.columns:
        st.header(f"Results for {col}:")

        correlations = df.corr()[[col]]
        st.subheader("Correlations without decay effect:")
        st.write(correlations)

        decayed_df = apply_decay(df, decay_rate)
        decayed_correlations = decayed_df.corr()[[col]]
        st.subheader("Correlations with decay effect:")
        st.write(decayed_correlations)

        time_to_90_percent_effect = find_time_to_reach_90_percent(decay_rate)
        st.subheader(f"Timeframe to reach 90% of the effect: {time_to_90_percent_effect} days")
        st.write("=" * 50)
