import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.import metrics

# Load the dataset
data = {
    'x1'=[22,5,6,8,10,11],
    'x2'=[1,3,4,9,11,14],
    'y'=[1.7,2.6,2.8,2.3,2.7,2.4]
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
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)
print("intercept:", model.intercept_)
print("coefficients:", model.coef_)
print("Mean absolute error:", metrics.mean_absolute_error(y_test, y_pred))
print("Mean squared error:", mse)
print("R-squared:", r2)