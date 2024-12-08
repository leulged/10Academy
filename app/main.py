import streamlit as st
from utils import load_data, plot_monthly_trends, plot_hourly_trends, plot_correlation_heatmap, plot_wind_solar_speed, detect_outliers, plot_outliers

# Title of the Streamlit app
st.title("Data Visualization Dashboard")

# Sidebar for user input (interactivity)
st.sidebar.header("User Input Features")

# Load the dataset
data_option = st.sidebar.selectbox('Select Dataset', ['Benin', 'Sierra Leone', 'Togo'])

# Load data dynamically based on user selection
data = load_data(data_option)

# Display the dataset
st.subheader(f"Dataset: {data_option}")
st.dataframe(data.head())

# Plotting options in the sidebar
chart_option = st.sidebar.selectbox('Select a chart to view', ['Monthly Trends', 'Hourly Trends', 'Correlation Heatmap', 'Wind & Solar Speed Analysis', 'Outlier Detection'])

if chart_option == 'Monthly Trends':
    plot_monthly_trends(data)
elif chart_option == 'Hourly Trends':
    plot_hourly_trends(data)
elif chart_option == 'Correlation Heatmap':
    plot_correlation_heatmap(data)
elif chart_option == 'Wind & Solar Speed Analysis':
    plot_wind_solar_speed(data)
elif chart_option == 'Outlier Detection':
    outliers = detect_outliers(data, ['GHI', 'DNI', 'DHI', 'WS'])
    plot_outliers(data, outliers)
