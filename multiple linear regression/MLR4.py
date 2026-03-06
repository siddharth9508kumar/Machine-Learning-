import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# Load the dataset
data = {
    'x1':[7,1,11,11,7,11,3],
    'x2':[2.6,2.9,5.6,3.1,5,2,5.5,7.1],
    'y' :[78.5,74.3,104.3,87.6,95.9,109.2,102.7]
}
df = pd.DataFrame(data)
# Define the independent variables (features) and the dependent variable (target)
X = df[['x1', 'x2']]
y = df['y']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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