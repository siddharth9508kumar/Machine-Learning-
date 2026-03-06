import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
# Load the dataset
data = {
    'x1':[1,2,3,4,5],
    'x2':[4,3,2,2,1],
    'y' :[2,4,3,6,5]
}
df = pd.DataFrame(data)
# Define the independent variables (features) and the dependent variable (target)
X = df[['x1', 'x2']]
y = df['y']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
# Create a Linear Regression model
model = LinearRegression()
# Fit the model to the training data
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model
print("intercept:", model.intercept_)
print("coefficients:", model.coef_)
print("Mean absolute error:", metrics.mean_absolute_error(y_test, y_pred))
print("Mean squared error:", metrics.mean_squared_error(y_test, y_pred))
print("R-squared:", metrics.r2_score(y_test, y_pred))