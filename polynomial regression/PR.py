import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
#data
x=np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y=np.array([45000,50000,60000,80000,110000,150000,200000,300000,500000,1000000])

degree=4
poly_features=PolynomialFeatures(degree=degree)
x_poly=poly_features.fit_transform(x)
m=LinearRegression()
m.fit(x_poly,y)
plt.scatter(x,y,color='red')
plt.plot(x,m.predict(x_poly),color='blue')
plt.title('Polynomial Regression')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

#comapare with linear regression
m2=LinearRegression()
m2.fit(x,y)
plt.scatter(x,y,color='red')
plt.plot(x,m2.predict(x),color='blue')
plt.title('Linear Regression')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

