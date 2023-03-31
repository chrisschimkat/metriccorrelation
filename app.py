import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Metrics Correlation")
st.markdown("See example input file [here](https://github.com/chrisschimkat/metriccorrelation/blob/main/Book1.csv)")

uploaded_file = st.file_uploader("Upload a CSV file:", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.capitalize() for col in df.columns]
    df.columns = [col.strip() for col in df.columns]  # Remove spaces in column names
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df.set_index('Date', inplace=True)

    def apply_decay(data, decay_rate):
        decayed_data = data.copy()
        for i in range(1, len(data)):
            decayed_data.iloc[i] = decayed_data.iloc[i] + decayed_data.iloc[i - 1] * decay_rate
        return decayed_data

    decay_rate = 0.9
    decayed_df = apply_decay(df, decay_rate)
    decayed_correlations = decayed_df.corr()

    st.header("Correlation matrix")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(decayed_correlations, annot=True, fmt='.2f', cmap='plasma_r', vmin=-1, vmax=1, ax=ax)
    ax.set_title('Correlations with decay effect')
    st.pyplot(fig)
    
    st.header("Top 10 correlations:")
    
    # Mask the lower triangle of the correlation matrix
    mask = np.triu(np.ones_like(decayed_correlations, dtype=bool), k=1)
    correlations_upper_triangle = decayed_correlations.where(mask)

    top_10_correlations = correlations_upper_triangle.stack().nlargest(10)
    st.write(top_10_correlations.to_frame('Correlation'))

    # Time series chart
    st.header("Time series chart for selected metrics")
    st.markdown("Select two metrics to see how they compare over time. Use this to help with identifying the timeframe between cause and effect.")
    selected_metrics = st.multiselect("Select two metrics to plot:", options=df.columns, default=df.columns[:2].tolist())

    if len(selected_metrics) == 2:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()

        ax1.plot(df.index, df[selected_metrics[0]], label=selected_metrics[0])
        ax2.plot(df.index, df[selected_metrics[1]], label=selected_metrics[1], color='orange')

        ax1.set_title('Time series chart for selected metrics')
        ax1.set_xlabel('Date')
        ax1.set_ylabel(selected_metrics[0])
        ax2.set_ylabel(selected_metrics[1])

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        st.pyplot(fig)
    else:
        st.warning("Please select exactly two metrics.")
