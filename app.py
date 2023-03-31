import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import requests
from io import StringIO

def read_csv_file(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.columns = [col.capitalize() for col in df.columns]
    df.columns = [col.strip() for col in df.columns]  # Remove spaces in column names
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df.set_index('Date', inplace=True)
    return df

st.title("Metrics Correlation")
# Add a button for loading the sample CSV
load_sample_csv = st.button('Load Sample CSV')

# Add a download link for the example input file
example_csv_link = '[here](https://raw.githubusercontent.com/chrisschimkat/metriccorrelation/main/Book1.csv?raw=true)'
st.markdown(f"See example input file {example_csv_link} (right-click and choose 'Save link as...' to download)")

uploaded_file = st.file_uploader("Upload a CSV file:", type=['csv'])

if load_sample_csv:
    sample_csv_url = 'https://raw.githubusercontent.com/chrisschimkat/metriccorrelation/main/Book1.csv'
    content = requests.get(sample_csv_url).content
    uploaded_file = StringIO(content.decode('utf-8'))

if uploaded_file is not None:
    df = read_csv_file(uploaded_file)
    st.session_state.df = df

if 'df' in st.session_state:
    df = st.session_state.df

    if 'correlations' not in st.session_state:
        correlations = df.corr()
        st.session_state.correlations = correlations
    else:
        correlations = st.session_state.correlations

    st.header("Correlation matrix")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, fmt='.2f', cmap='plasma_r', vmin=-1, vmax=1, ax=ax)
    ax.set_title('Correlations')
    st.pyplot(fig)

    # Save the plot to a buffer
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)

    # Convert the buffer to bytes
    img_bytes = buffer.getvalue()
    buffer.close()

    # Export heatmap plot to PNG
    if st.button('Export heatmap plot to PNG'):
        st.download_button("Download heatmap plot", img_bytes, "heatmap_plot.png", "image/png")
    
    st.header("Top 10 correlations:")
    
    # Mask the lower triangle of the correlation matrix
    mask = np.triu(np.ones_like(correlations, dtype=bool), k=1)
    correlations_upper_triangle = correlations.where(mask)

    top_10_correlations = correlations_upper_triangle.stack().nlargest(10)
    st.write(top_10_correlations.to_frame('Correlation'))

   # Time series chart
    st.header("Time series chart for selected metrics")
    st.markdown("Select two metrics to see how they compare over time. Use this to help with identifying the timeframe between cause and effect.")

    if "selected_metrics" not in st.session_state:
        st.session_state.selected_metrics = df.columns[:2].tolist()

    selected_metrics = st.multiselect("Select two metrics to plot:", options=df.columns, default=st.session_state.selected_metrics)

    if len(selected_metrics) == 2:
        st.session_state.selected_metrics = selected_metrics

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df[selected_metrics[0]], label=selected_metrics[0])
        ax.set_ylabel(selected_metrics[0], fontsize=12)

        ax2 = ax.twinx()
        ax2.plot(df[selected_metrics[1]], color='orange', label=selected_metrics[1])
        ax2.set_ylabel(selected_metrics[1], fontsize=12)

        ax.set_xlabel('Date', fontsize=12)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

        st.pyplot(fig)
        
        # Save the plot to a buffer
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)

        # Convert the buffer to bytes
        img_bytes = buffer.getvalue()
        buffer.close()

        # Export time series plot to PNG
        if st.button('Export time series plot to PNG'):
            st.download_button("Download time series plot", img_bytes, "time_series_plot.png", "image/png")
    else:
        st.warning("Please select exactly two metrics.")

