import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy.stats import zscore

def load_data(dataset_name):
    """
    Function to load data based on user selection.
    """
    if dataset_name == 'Benin':
        df = pd.read_csv('files/benin-malanville.csv')  # Correct path
    elif dataset_name == 'Sierra Leone':
        df = pd.read_csv('files/sierraleone-bumbuna.csv')  # Correct path
    elif dataset_name == 'Togo':
        df = pd.read_csv('files/togo-dapaong_qc.csv')  # Correct path
    
    # Handle missing values and preprocessing if needed
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.dropna(subset=['GHI', 'DNI', 'DHI', 'Tamb'])
    
    return df


def plot_monthly_trends(df):
    """
    Function to plot monthly trends for GHI, DNI, DHI, and Tamb.
    """
    monthly_data = df.groupby(df['Timestamp'].dt.month)[['GHI', 'DNI', 'DHI', 'Tamb']].mean()
    monthly_data.plot(kind='line', figsize=(10, 6), marker='o')
    plt.title('Monthly Trends for GHI, DNI, DHI, and Tamb')
    plt.xlabel('Month')
    plt.ylabel('Values')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    st.pyplot()  # Render the plot in Streamlit

def plot_hourly_trends(df):
    """
    Function to plot hourly trends for GHI, DNI, DHI, and Tamb.
    """
    hourly_data = df.groupby(df['Timestamp'].dt.hour)[['GHI', 'DNI', 'DHI', 'Tamb']].mean()
    hourly_data.plot(kind='line', figsize=(10, 6), marker='o')
    plt.title('Hourly Trends for GHI, DNI, DHI, and Tamb')
    plt.xlabel('Hour')
    plt.ylabel('Values')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    st.pyplot()  # Render the plot in Streamlit

def plot_correlation_heatmap(df):
    """
    Function to plot the correlation heatmap for the selected dataset.
    """
    correlation_matrix = df[['GHI', 'DNI', 'DHI', 'Tamb']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix: Solar Radiation and Temperature")
    plt.show()
    st.pyplot()  # Render the plot in Streamlit

# Wind and Solar Speed Analysis
def plot_wind_solar_speed(df):
    """
    Function to plot wind speed and solar irradiance (GHI, DNI, DHI).
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    # Plot Wind Speed vs Wind Direction (WD)
    ax[0].scatter(df['WD'], df['WS'], c='blue', alpha=0.6)
    ax[0].set_title('Wind Speed vs Wind Direction')
    ax[0].set_xlabel('Wind Direction (Degrees)')
    ax[0].set_ylabel('Wind Speed (m/s)')
    ax[0].grid(True)

    # Plot Solar Irradiance (GHI, DNI, DHI)
    ax[1].plot(df['Timestamp'], df['GHI'], label='GHI', color='orange')
    ax[1].plot(df['Timestamp'], df['DNI'], label='DNI', color='green')
    ax[1].plot(df['Timestamp'], df['DHI'], label='DHI', color='red')
    ax[1].set_title('Solar Irradiance Over Time')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Irradiance (W/m^2)')
    ax[1].legend(loc='best')
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()
    st.pyplot()  # Render the plot in Streamlit

# Outlier Detection
def detect_outliers(df, columns):
    """
    Function to detect outliers based on Z-score.
    """
    z_scores = df[columns].apply(zscore)
    outliers = (z_scores.abs() > 3)
    return outliers

def plot_outliers(df, outliers):
    """
    Function to plot outliers detected in the dataset.
    """
    outlier_data = df[outliers.any(axis=1)]
    st.write(f"Outliers Detected: {outlier_data.shape[0]}")
    st.dataframe(outlier_data)  # Show the detected outliers

    # Plot outliers
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Timestamp'], df['GHI'], color='blue', label='Normal Data')
    plt.scatter(outlier_data['Timestamp'], outlier_data['GHI'], color='red', label='Outliers')
    plt.title('Outliers Detection - GHI')
    plt.xlabel('Timestamp')
    plt.ylabel('GHI (W/m^2)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
    st.pyplot()  # Render the plot in Streamlit

import os
print("Current Working Directory: ", os.getcwd())  # To check the current working directory
