import pandas as pd

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
