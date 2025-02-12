import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = load_diabetes()
print(data.keys())

print(data.DESCR)

df = data ['frame']
print(df)

plt.hist(df["target"], 25)
plt.xlabel("target")
plt.show()

sns.heatmap(df.corr(), round(2) ,annot=True)
plt.show()

plt.subplot(1,2,1)
plt.scatter(df['bmi'], df['target'])
plt.xlabel("bmi")
plt.ylabel("target")
plt.subplot(1,2,2)
plt.scatter(df['s5'], df['target'])
plt.xlabel('s5')
plt.ylabel('target')
plt.show()

x= pd.DataFrame(df[['bmi', 's5']], columns=['bmi', 's5'])
y= df['target']

print(x)
print(y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 5)
print(X_train.shape)
print(X_test.shape)
lm = LinearRegression()
lm.fit(X_train, y_train)
y_train_predict  = lm.predict(X_train)

rmse =  (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)
print(f"RMSE (test) = {rmse_test}, R2 (test)= {r2_test}")
print(X_train, y_test_predict)

