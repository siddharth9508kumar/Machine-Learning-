from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
x=np.array([[-1],[0],[1],[2]])
y=np.array([0,2,4,5])
model=LinearRegression()
model.fit(x,y)
print("Coefficients:")
print("b0:",model.intercept_)
print("b1:",model.coef_[0])
# to test the model
y_pred=model.predict(x)
print("Predicted values:",y_pred)
# to evaluate the model
mae=mean_absolute_error(y,y_pred)
mse=mean_squared_error(y,y_pred)
r2=r2_score(y,y_pred)
print("Mean Absolute Error:",mae)
print("Mean Squared Error:",mse)
print("R-squared:",r2)