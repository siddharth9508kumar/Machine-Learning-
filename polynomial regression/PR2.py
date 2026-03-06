import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
#  dataset
data=pd.read_csv('student_scores.csv')
print(data.head(10))
x=data[['G1','G2']]
y=data['G3']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#FOR LINEAR REGRESSION
linear_model=LinearRegression()
linear_model.fit(x_train,y_train)
print("Linear Regression Coefficients:", linear_model.coef_)
print("Linear Regression Intercept:", linear_model.intercept_)
y_pred_linear=linear_model.predict(x_test)
print("Linear Regression Mean Squared Error:", mean_squared_error(y_test, y_pred_linear))
print("Linear Regression R^2 Score:", r2_score(y_test, y_pred_linear))
plt.figure()
plt.scatter(x_test['G1'], y_test, color='red', label='Actual G3')
plt.plot(x_test['G1'], y_pred_linear, color='blue', label='Predicted G3 (Linear)')
plt.title('Linear Regression: G1 vs G3')
plt.xlabel('Actual G1')
plt.ylabel('Predicted G3')
plt.show()
#FOR POLYNOMIAL REGRESSION
degree=2
poly_features=PolynomialFeatures(degree=degree)
x_poly=poly_features.fit_transform(x)
x_train_poly, x_test_poly, y_train_poly, y_test_poly = train_test_split(x_poly, y, test_size=0.2, random_state=0)
poly_model=LinearRegression()
poly_model.fit(x_train_poly, y_train_poly)
print("Polynomial Regression Coefficients:", poly_model.coef_)
y_pred_poly=poly_model.predict(x_test_poly)
print("Polynomial Regression Mean Squared Error:", mean_squared_error(y_test_poly, y_pred_poly))
print("Polynomial Regression R^2 Score:", r2_score(y_test_poly, y_pred_poly))   
plt.figure()
plt.scatter(x_test['G1'], y_test, color='red', label='Actual G3')
plt.scatter(x_test['G1'], y_pred_poly, color='blue', label='Predicted G3 (Polynomial)')
plt.title('Polynomial Regression: G1 vs G3')
plt.xlabel('Actual G1')
plt.ylabel('Predicted G3')
plt.show()

# Compare Linear and Polynomial Regression
plt.figure()
plt.scatter(x_test['G1'], y_test, color='red', label='Actual G3')
plt.scatter(x_test['G1'], y_pred_linear, color='blue', label='Predicted G3 (Linear)')
plt.scatter(x_test['G1'], y_pred_poly, color='green', label='Predicted G3 (Polynomial)')
plt.title('Comparison of Linear and Polynomial Regression: G1 vs G3')
plt.xlabel('Actual G1')
plt.ylabel('Predicted G3')
plt.legend()
plt.show()

