import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from windrose import WindroseAxes
import numpy as np

from matplotlib import cm  # Import Matplotlib's colormap library


# Load CSV files
benin_df = pd.read_csv('../files/benin-malanville.csv')
sierra_df = pd.read_csv('../files/sierraleone-bumbuna.csv')
togo_df = pd.read_csv('../files/togo-dapaong_qc.csv')

# Inspect the data
print("Benin Data:")
print(benin_df.head())
print(benin_df.info())  # Information about Benin dataset

print("\nSierra Leone Data:")
print(sierra_df.head())
print(sierra_df.info())  # Information about Sierra Leone dataset

print("\nTogo Data:")
print(togo_df.head())
print(togo_df.info())  # Information about Togo dataset

# Descriptive statistics for all three datasets
print("\nBenin Data Descriptive Statistics:")
print(benin_df.describe())
print("\nSierra Leone Data Descriptive Statistics:")
print(sierra_df.describe())
print("\nTogo Data Descriptive Statistics:")
print(togo_df.describe())

# Dropping the 'Comments' column if it's unnecessary
benin_df = benin_df.drop(columns=['Comments'], errors='ignore')  # errors='ignore' to avoid errors if 'Comments' doesn't exist
sierra_df = sierra_df.drop(columns=['Comments'], errors='ignore')
togo_df = togo_df.drop(columns=['Comments'], errors='ignore')

# Check for missing values after dropping 'Comments' column
print("\nBenin Data - Missing Values:")
print(benin_df.isnull().sum())  # Show missing values for Benin dataset

print("\nSierra Leone Data - Missing Values:")
print(sierra_df.isnull().sum())  # Show missing values for Sierra Leone dataset

print("\nTogo Data - Missing Values:")
print(togo_df.isnull().sum())  # Show missing values for Togo dataset

# Handle missing values by filling with the median (if necessary)
## Fill missing values with the median only for numeric columns
benin_df_numeric = benin_df.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
benin_df[benin_df_numeric.columns] = benin_df_numeric.fillna(benin_df_numeric.median())

sierra_df_numeric = sierra_df.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
sierra_df[sierra_df_numeric.columns] = sierra_df_numeric.fillna(sierra_df_numeric.median())

togo_df_numeric = togo_df.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
togo_df[togo_df_numeric.columns] = togo_df_numeric.fillna(togo_df_numeric.median())


# Verify that missing values have been handled
print("\nBenin Data - Missing Values After Handling:")
print(benin_df.isnull().sum())  # Verify missing values for Benin

print("\nSierra Leone Data - Missing Values After Handling:")
print(sierra_df.isnull().sum())  # Verify missing values for Sierra Leone

print("\nTogo Data - Missing Values After Handling:")
print(togo_df.isnull().sum())  # Verify missing values for Togo
# Check if 'Comments' column exists in Benin dataframe
print("'Comments' in Benin Data:", 'Comments' in benin_df.columns)

# Check if 'Comments' column exists in Sierra Leone dataframe
print("'Comments' in Sierra Leone Data:", 'Comments' in sierra_df.columns)

# Check if 'Comments' column exists in Togo dataframe
print("'Comments' in Togo Data:", 'Comments' in togo_df.columns)

benin_df['Timestamp'] = pd.to_datetime(benin_df['Timestamp'])

# Extract time components
benin_df['Month'] = benin_df['Timestamp'].dt.month
benin_df['Day'] = benin_df['Timestamp'].dt.day
benin_df['Hour'] = benin_df['Timestamp'].dt.hour

# Aggregate data by month
monthly_data = benin_df.groupby('Month')[['GHI', 'DNI', 'DHI', 'Tamb']].mean()

# Plot line chart for monthly trends
monthly_data.plot(kind='line', figsize=(10, 6), marker='o')
plt.title('Monthly Trends for GHI, DNI, DHI, and Tamb')
plt.xlabel('Month')
plt.ylabel('Values')
plt.legend(loc='best')
plt.grid()
plt.show()

# Aggregate data by hour
hourly_data = benin_df.groupby('Hour')[['GHI', 'DNI', 'DHI', 'Tamb']].mean()

# Plot line chart for hourly patterns
hourly_data.plot(kind='line', figsize=(10, 6), marker='o')
plt.title('Hourly Trends for GHI, DNI, DHI, and Tamb')
plt.xlabel('Hour')
plt.ylabel('Values')
plt.legend(loc='best')
plt.grid()
plt.show()

# Group data by cleaning events
cleaning_impact = benin_df.groupby('Cleaning')[['ModA', 'ModB']].mean()

# Plot the impact of cleaning
cleaning_impact.plot(kind='bar', figsize=(8, 5), color=['blue', 'orange'])
plt.title('Impact of Cleaning on Sensor Readings (ModA and ModB)')
plt.xlabel('Cleaning Event')
plt.ylabel('Sensor Values')
plt.legend(loc='best')
plt.grid()
plt.show()


# Convert 'Timestamp' column to datetime
sierra_df['Timestamp'] = pd.to_datetime(sierra_df['Timestamp'])

# Extract time components
sierra_df['Month'] = sierra_df['Timestamp'].dt.month
sierra_df['Day'] = sierra_df['Timestamp'].dt.day
sierra_df['Hour'] = sierra_df['Timestamp'].dt.hour

# Aggregate data by month
monthly_data_sierra = sierra_df.groupby('Month')[['GHI', 'DNI', 'DHI', 'Tamb']].mean()

# Plot line chart for monthly trends
monthly_data_sierra.plot(kind='line', figsize=(10, 6), marker='o')
plt.title('Sierra Leone: Monthly Trends for GHI, DNI, DHI, and Tamb')
plt.xlabel('Month')
plt.ylabel('Values')
plt.legend(loc='best')
plt.grid()
plt.show()

# Aggregate data by hour
hourly_data_sierra = sierra_df.groupby('Hour')[['GHI', 'DNI', 'DHI', 'Tamb']].mean()

# Plot line chart for hourly patterns
hourly_data_sierra.plot(kind='line', figsize=(10, 6), marker='o')
plt.title('Sierra Leone: Hourly Trends for GHI, DNI, DHI, and Tamb')
plt.xlabel('Hour')
plt.ylabel('Values')
plt.legend(loc='best')
plt.grid()
plt.show()

# Group data by cleaning events
cleaning_impact_sierra = sierra_df.groupby('Cleaning')[['ModA', 'ModB']].mean()

# Plot the impact of cleaning
cleaning_impact_sierra.plot(kind='bar', figsize=(8, 5), color=['blue', 'orange'])
plt.title('Sierra Leone: Impact of Cleaning on Sensor Readings (ModA and ModB)')
plt.xlabel('Cleaning Event')
plt.ylabel('Sensor Values')
plt.legend(loc='best')
plt.grid()
plt.show()



# Convert 'Timestamp' column to datetime
togo_df['Timestamp'] = pd.to_datetime(togo_df['Timestamp'])

# Extract time components
togo_df['Month'] = togo_df['Timestamp'].dt.month
togo_df['Day'] = togo_df['Timestamp'].dt.day
togo_df['Hour'] = togo_df['Timestamp'].dt.hour

# Aggregate data by month
monthly_data_togo = togo_df.groupby('Month')[['GHI', 'DNI', 'DHI', 'Tamb']].mean()

# Plot line chart for monthly trends
monthly_data_togo.plot(kind='line', figsize=(10, 6), marker='o')
plt.title('Togo: Monthly Trends for GHI, DNI, DHI, and Tamb')
plt.xlabel('Month')
plt.ylabel('Values')
plt.legend(loc='best')
plt.grid()
plt.show()


# Aggregate data by hour
hourly_data_togo = togo_df.groupby('Hour')[['GHI', 'DNI', 'DHI', 'Tamb']].mean()

# Plot line chart for hourly patterns
hourly_data_togo.plot(kind='line', figsize=(10, 6), marker='o')
plt.title('Togo: Hourly Trends for GHI, DNI, DHI, and Tamb')
plt.xlabel('Hour')
plt.ylabel('Values')
plt.legend(loc='best')
plt.grid()
plt.show()

# Group data by cleaning events
cleaning_impact_togo = togo_df.groupby('Cleaning')[['ModA', 'ModB']].mean()

# Plot the impact of cleaning
cleaning_impact_togo.plot(kind='bar', figsize=(8, 5), color=['blue', 'orange'])
plt.title('Togo: Impact of Cleaning on Sensor Readings (ModA and ModB)')
plt.xlabel('Cleaning Event')
plt.ylabel('Sensor Values')
plt.legend(loc='best')
plt.grid()
plt.show()



# Select relevant columns
correlation_cols = ['GHI', 'DNI', 'DHI', 'TModA', 'TModB']
correlation_matrix = benin_df[correlation_cols].corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix: Solar Radiation and Temperature")
plt.show()
# Plot pair plots for solar radiation and temperature measures
sns.pairplot(benin_df[correlation_cols], diag_kind="kde", corner=True)
plt.suptitle("Pair Plot: Solar Radiation and Temperature", y=1.02)
plt.show()
# Select relevant columns for wind and solar irradiance
wind_solar_cols = ['WS', 'WSgust', 'WD', 'GHI', 'DNI', 'DHI']

# Pair plot to visualize scatter relationships
sns.pairplot(benin_df[wind_solar_cols], diag_kind="kde", corner=True)
plt.suptitle("Scatter Matrix: Wind Conditions and Solar Irradiance", y=1.02)
plt.show()

# For Sierra Leone
correlation_matrix_sierra = sierra_df[correlation_cols].corr()
sns.heatmap(correlation_matrix_sierra, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Sierra Leone: Correlation Matrix")
plt.show()

# For Togo
correlation_matrix_togo = togo_df[correlation_cols].corr()
sns.heatmap(correlation_matrix_togo, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Togo: Correlation Matrix")
plt.show()


# Convert WD (wind direction) to numeric if needed
benin_df['WD'] = pd.to_numeric(benin_df['WD'], errors='coerce')

# Bin wind direction into 8 compass directions (N, NE, E, SE, S, SW, W, NW)
wind_bins = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
benin_df['Wind Bin'] = pd.cut(
    benin_df['WD'], bins=np.linspace(0, 360, 9), labels=wind_bins, include_lowest=True
)

# Calculate mean wind speed for each bin
wind_analysis = benin_df.groupby('Wind Bin', observed=False)['WS'].mean()

# Radial plot
angles = np.linspace(0, 2 * np.pi, len(wind_bins), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

wind_analysis = wind_analysis.tolist()
wind_analysis += wind_analysis[:1]  # Complete the circle

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.bar(angles, wind_analysis, width=0.4, color='blue', alpha=0.6, edgecolor='black')

# Add labels
ax.set_yticks([1, 2, 3, 4])  # Adjust to your scale
ax.set_xticks(angles[:-1])
ax.set_xticklabels(wind_bins)
ax.set_title('Radial Bar Plot of Wind Speed and Direction', va='bottom')
plt.show()


# # # Create a wind rose plot
# # fig = plt.figure(figsize=(8, 6))
# # ax = WindroseAxes.from_ax(fig=fig)
# # ax.bar(benin_df['WD'], benin_df['WS'], normed=True, opening=0.8, edgecolor='black', cmap='viridis')

# # Add labels and title
# ax.set_title("Wind Rose: Wind Speed and Direction")
# ax.set_legend(title="Wind Speed (m/s)")
# plt.show()

# wind_direction_std = benin_df['WD'].std()
# print("Standard Deviation of Wind Direction:", wind_direction_std)

# plt.figure(figsize=(10, 6))
# plt.plot(benin_df['Timestamp'], benin_df['WD'], color='blue', alpha=0.7)
# plt.title("Wind Direction Variability Over Time")
# plt.xlabel("Time")
# plt.ylabel("Wind Direction (degrees)")
# plt.grid()
# plt.show()
# # Create a wind rose plot
# fig = plt.figure(figsize=(8, 6))
# ax = WindroseAxes.from_ax(fig=fig)
# ax.bar(
#     benin_df['WD'], 
#     benin_df['WS'], 
#     normed=True, 
#     opening=0.8, 
#     edgecolor='black', 
#     cmap=cm.get_cmap('viridis')  # Pass colormap object instead of a string
# )

# # Add labels and title
# ax.set_title("Wind Rose: Wind Speed and Direction")
# ax.set_legend(title="Wind Speed (m/s)")
# plt.show()

