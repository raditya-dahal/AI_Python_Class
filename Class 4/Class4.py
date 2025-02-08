# import numpy as np
# import matplotlib.pyplot as plt
# data = np.loadtxt('linreg_data.csv', delimiter=',', skiprows=1)
# x = data[:, 0]
# y = data[:, 1]
# slope, intercept = np.polyfit(x, y, 1)
# def myfunc(x):
#     return slope * x + intercept
# mymodel = myfunc(x)
# plt.scatter(x, y, label='Data Points', color='cyan')
# plt.plot(x, mymodel, color='red', label=f'Fit: y = {slope:.2f}x + {intercept:.2f}')
#
# plt.show()







# import numpy as np
# import matplotlib.pyplot as plt
# data = np.loadtxt('linreg_data.csv', delimiter=',')
# x = data[:, 0]
# y = data[:, 1]
# slope, intercept = np.polyfit(x, y, 1)
# mymodel = slope * x + intercept
# plt.scatter(x, y, label='Data Points')
# plt.plot(x, mymodel, color='red', label=f'Fit: y = {slope:.2f}x + {intercept:.2f}')
# plt.show()





# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn import linear_model
#
# data = np.genfromtxt("linreg_data.csv", delimiter=",")
# print(data)
# xp = data[:,0]
# yp = data[:,1]
# print(xp)
# xp = xp.reshape(-1,1)
# yp = yp.reshape(-1,1)
# print("xp=", xp)
#
# regr = linear_model.LinearRegression()
# regr.fit(xp, yp)
#
# print("slope=", regr.coef_)
# print("yintercept=", regr.intercept_)






# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib
# from sklearn import linear_model
# from sklearn import metrics
# my_data = np.genfromtxt("linreg_data.csv",delimiter=",")
# print(my_data)
# xp = my_data[:,0]
# yp = my_data[:,1]
# print(xp)
# xp = xp.reshape(-1,1)
# yp = yp.reshape(-1,1)
# print("xp =",xp)
#
# regr = linear_model.LinearRegression()
#
# # training/fitting model with training data
# regr.fit(xp,yp)
# print("slope b =", regr.coef_)
# print("y-intercept a=",regr.intercept_)
#
# #calculating prediction
# xval = np.full((1,1),0.5)
# yval = regr.predict(xval)
# print(yval)
# yhat = regr.predict(xp)
# print("yhat =",yhat)
#
# #Evaluation
# print("mean Absolute Error (MAE):",metrics.mean_absolute_error(yp, yhat))
# print("Mean Squared Error (MSE):", metrics.mean_squared_error(yp, yhat))
# print("Root mean squared Error (RMSE) :",np.sqrt(metrics.mean_squared_error(yp, yhat)))
# print("R2 value:", metrics.r2_score(yp,yhat))


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# data = pd.read_csv("linreg_data.csv", header=None)
#
#
# data.columns = ['x', 'y']
#
#
# print("First few rows of data:\n", data.head())
#
#
# x = data['x'].values
# y = data['y'].values
#
#
# coefficients = np.polyfit(x, y, 2)
# a, b, c = coefficients
#
# print(f"Quadratic Model: y = {a:.3f}x^2 + {b:.3f}x + {c:.3f}")
#
#
# y_fit = a * x**2 + b * x + c
#
#
# plt.scatter(x, y, color='blue', label='Data Points')
# plt.plot(x, y_fit, color='red', label='Fitted Quadratic Curve')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.title("Quadratic Fit to Data")
# plt.show()
#
#
#
# import numpy as np
# import matplotlib.pyplot as plt
# data = np.loadtxt('linreg_data.csv', delimiter=',')
# x, y = data[:, 0], data[:, 1]
# degree = 2
# p = np.poly1d(np.polyfit(x, y, degree))
# plt.scatter(x, y)
# plt.plot(np.sort(x), p(np.sort(x)), color='red')
# plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def polynomial_regression(file_path, degree=2):
    # Load data
    data = pd.read_csv(file_path, header=None, names=['x', 'y'])

    # Extract x and y values
    x = data[['x']].values
    y = data['y'].values

    # Transform features for polynomial regression
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x)

    # Fit polynomial regression model
    model = LinearRegression()
    model.fit(x_poly, y)

    # Generate fitted values
    y_fit = model.predict(x_poly)

    # Print model equation
    coefficients = model.coef_
    intercept = model.intercept_
    equation_terms = [f"{coef:.3f}x^{i}" if i > 0 else f"{coef:.3f}"
                      for i, coef in enumerate(coefficients)]
    equation = " + ".join(equation_terms)
    print(f"Polynomial Model (degree {degree}): y = {intercept:.3f} + {equation}")

    # Plot data and fitted curve
    plt.scatter(x, y, color='blue', label='Data Points')
    plt.plot(x, y_fit, color='red', label=f'Fitted Polynomial (degree {degree})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(f"Polynomial Regression (degree {degree})")
    plt.show()
polynomial_regression("linreg_data.csv", degree=2)




