from sklearn.datasets import fetch_california_housing 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error 
import pandas as pd 

# Load California housing data 
housing_data = fetch_california_housing() 
df = pd.DataFrame(housing_data.data, columns=housing_data.feature_names) 
df['MedianPrice'] = housing_data.target 

# Preview the dataset 
print(df.head()) 
print(f"\nTotal records: {len(df)}") 

# Prepare features and target 
X = df.drop('MedianPrice', axis=1) 
y = df['MedianPrice'] 

# Split the data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Create and train the regression model 
regressor = LinearRegression() 
regressor.fit(X_train, y_train) 

# Evaluate the model 
y_pred = regressor.predict(X_test) 
mae = mean_absolute_error(y_test, y_pred) 
print(f"\nMean Absolute Error: ${mae * 100000:.0f}")