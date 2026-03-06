from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
x=np.array([[1],[3],[7],[10],[12],[13],[16],[18],[20]])
y=np.array([4,5,5,8,7,6,9,12,11])
model=LinearRegression()
model.fit(x,y)
print("Slope (b1):",model.coef_[0])
print("Intercept (b0):",model.intercept_)
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