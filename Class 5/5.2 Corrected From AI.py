import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
data = load_diabetes()
print(data.keys())  # Check available keys

# Create a DataFrame manually
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add target column to the DataFrame
df['target'] = data.target

# Check if the DataFrame is loaded correctly
print(df.head())

# Plot histogram of the target variable
plt.hist(df["target"], bins=25)
plt.xlabel("target")
plt.show()

# Heatmap of correlations
sns.heatmap(df.corr().round(2), annot=True)
plt.show()

# Scatter plots
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(df['bmi'], df['target'])
plt.xlabel("bmi")
plt.ylabel("target")

plt.subplot(1, 2, 2)
plt.scatter(df['s5'], df['target'])
plt.xlabel('s5')
plt.ylabel('target')

plt.show()

# Selecting features and target
x = df[['bmi', 's5']]
y = df['target']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

# Train the model
lm = LinearRegression()
lm.fit(X_train, y_train)

# Predicting on training data
y_train_predict = lm.predict(X_train)

# Predicting on test data
y_test_predict = lm.predict(X_test)

# Calculate error metrics
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_predict))
r2_train = r2_score(y_train, y_train_predict)

rmse_test = np.sqrt(mean_squared_error(y_test, y_test_predict))
r2_test = r2_score(y_test, y_test_predict)

# Print results
print(f"RMSE (train) = {rmse_train}, R2 (train) = {r2_train}")
print(f"RMSE (test) = {rmse_test}, R2 (test) = {r2_test}")
