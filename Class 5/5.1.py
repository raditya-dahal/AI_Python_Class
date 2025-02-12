#Correlation, Import any CSV file and print with correlation

import pandas as pd

# Load CSV file
df = pd.read_csv("weight-height.csv")  # Replace with your actual file

# Print the first few rows
print(df.head())

# Select only numeric columns for correlation
df_numeric = df.select_dtypes(include=['number'])

# Compute and print correlation
print("\nCorrelation Matrix:")
print(df_numeric.corr())

