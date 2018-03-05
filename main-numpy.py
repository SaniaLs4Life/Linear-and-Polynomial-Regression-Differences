import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data.csv")

X = data["x"]
Y = data["y"]

X = np.array(X)
Y = np.array(Y)

a,b,c = np.polyfit(X,Y,2)

x = np.arange(2200)

plt.scatter(X,Y, color="red")
plt.plot(x, a*x**2+b*x+c, color="blue")
plt.show()