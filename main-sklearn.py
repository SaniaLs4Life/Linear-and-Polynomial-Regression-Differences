import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures as pf
from sklearn.linear_model import LinearRegression as lr

data = pd.read_csv("data.csv")

X = data["x"]
Y = data["y"]

X = np.array(X).reshape(len(X), 1)
Y = np.array(Y).reshape(len(Y), 1)

lrg = lr()
lrg.fit(X,Y)
plt.scatter(X,Y)
plt.show()
predictX = lrg.predict(X)
plt.plot(X, predictX, color="red")


plt.scatter(X,Y, color="blue")
pl = pf(degree=2)
X_New = pl.fit_transform(X)

lrg2 = lr()
lrg2.fit(X_New, Y)
predictX2 = lrg2.predict(X_New)

plt.title("Linear and Polynomial Regression")
plt.plot(X, predictX2, color="orange")
plt.show()






