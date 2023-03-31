import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
import requests
from io import StringIO
from streamlit.hashing import _CodeHasher
import streamlit.ReportThread as ReportThread
from streamlit.server.Server import Server

class SessionState(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def get_session():
    session = None
    ctx = ReportThread.get_report_ctx()
    this_session = None

    current_server = Server.get_current()
    if hasattr(current_server, '_session_infos'):
        # Streamlit >= 0.84
        session_infos = Server.get_current()._session_infos.values()
    else:
        session_infos = Server.get_current()._session_info_by_id.values()

    for session_info in session_infos:
        s = session_info.session
        if (
            # Streamlit < 0.84.0
            (hasattr(s, '_main_dg') and s._main_dg == ctx.main_dg)
            # Streamlit >= 0.84.0
            or (not hasattr(s, '_main_dg') and s.enqueue == ctx.enqueue)
        ):
            session = s
            break

    if session is not None:
        if not hasattr(session, '_custom_session_state'):
            session._custom_session_state = SessionState(**kwargs)
        this_session = session._custom_session_state

    return this_session

st.title("Metrics Correlation")
# Add a button for loading the sample CSV
load_sample_csv = st.button('Load Sample CSV')

# Add a download link for the example input file
example_csv_link = '[here](https://raw.githubusercontent.com/chrisschimkat/metriccorrelation/main/Book1.csv?raw=true)'
st.markdown(f"See example input file {example_csv_link} (right-click and choose 'Save link as...' to download)")

state = get_session()
if 'df' not in state:
    state.df = None

uploaded_file = st.file_uploader("Upload a CSV file:", type=['csv'])

if load_sample_csv:
    sample_csv_url = 'https://raw.githubusercontent.com/chrisschimkat/metriccorrelation/main/Book1.csv'
    content = requests.get(sample_csv_url).content
    uploaded_file = StringIO(content.decode('utf-8'))

if uploaded_file is not None:
    state.df = pd.read_csv(uploaded_file)
    state.df.columns = [col.capitalize() for col in state.df.columns]
    state.df.columns = [col.strip() for col in state.df.columns]  # Remove spaces in column names
    state.df['Date'] = pd.to_datetime(state.df['Date'], dayfirst=True)
    state.df.set_index('Date', inplace=True)

if state.df is not None:
    df = state.df

    # Calculate correlations and time lags for all pairs of series
    corr_values = []
    time_lags = []
    for i, series1 in enumerate(df.columns):
        for j, series2 in enumerate(df.columns):
            if j <= i:
                continue
            corr = df[series1].corr(df[series2])
            lag_range = np.arange(-30, 31)  # Range of time lags to consider
            max_corr = -1
            max_lag = 0
            for lag in lag_range:
                corr_lag = df[series1].corr(df[series2].shift(lag))
                if corr_lag > max_corr:
                    max_corr = corr_lag
                    max_lag = lag
            corr_values.append(corr)
            time_lags.append(max_lag)

    # Create a dataframe of the top 10 correlated metrics with time lags
    corr_df = pd.DataFrame({
        'Series 1': [],
        'Series 2': [],
        'Correlation': [],
        'Time lag (days)': []
    })

    # Find the top 10 correlations
    indices = np.argsort(corr_values)[-10:]
    for idx in indices:
        series1 = df.columns[idx // len(df.columns)]
        series2 = df.columns[idx % len(df.columns)]
        correlation = corr_values[idx]
        time_lag = time_lags[idx]
        corr_df = corr_df.append({
            'Series 1': series1,
            'Series 2': series2,
            'Correlation': correlation,
            'Time lag (days)': time_lag
        }, ignore_index=True)

    # Sort the table by correlation value
    corr_df = corr_df.sort_values(by='Correlation', ascending=False)

    # Show the table of top 10 correlated metrics with time lags
    st.header("Time lags between top 10 correlated metrics")
    st.write(corr_df)

