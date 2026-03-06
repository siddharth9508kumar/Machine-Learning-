import numpy as np
x=np.array([1,2,3,4,5])
y=np.array([2,4,5,4,5])
x_mean=np.mean(x)
y_mean=np.mean(y)
numerator=np.sum((x-x_mean)*(y-y_mean))
denominator=np.sum((x-x_mean)**2)
b1=numerator/denominator
b0=y_mean-b1*x_mean
print("Coefficients:")
print("b0:",b0)
print("b1:",b1)
# to test the model
y_pred=b0+b1*x
print("Predicted values:",y_pred)