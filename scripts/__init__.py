
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('path/to/dataset.csv')

# Summary statistics
print(df.describe())

# Missing values check
print(df.isnull().sum())

# Plot solar radiation trends
plt.plot(df['Timestamp'], df['GHI'], label='GHI')
plt.xlabel('Timestamp')
plt.ylabel('GHI')
plt.legend()
plt.show()
