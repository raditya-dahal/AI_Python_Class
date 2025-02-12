import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df= pd.read_csv('diamonds.csv')
print(df.head())

x = df[['carat', 'depth', 'table', 'x', 'y','z']]
y = df[['price']]
print(x,y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=101)

alphas = [0.1,0.2,0.3,0.4,0.5,1,2,3,4,5,6,7,8]
scores = []
for alp in alphas:
    lasso = linear_model.Lasso(alpha=alp)
    lasso.fit(x_train, y_train)
    print(lasso.coef_.round(2), lasso.intercept_)
    sc = lasso.score(x_test, y_test)
    scores.append(sc)
    print("alpha=", alp," lasso score:", sc)

plt.plot(alphas, scores)
plt.show()

best_r2 = max(scores)
idx = scores.index(best_r2)
best_alpha = alphas[idx]

print(f"\nBest R2 = {best_r2}, Best alpha = {best_alpha}")
