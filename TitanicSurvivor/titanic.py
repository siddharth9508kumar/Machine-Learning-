# Import libraries 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 

# Load the Titanic training data 
titanic_df = pd.read_csv('titanic.csv') 

# Explore the dataset 
print(titanic_df.head()) 
print(f"\nMissing values per column:\n{titanic_df.isnull().sum()}") 

# Handle missing data 
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].median()) 
titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S') 

# Select features for prediction 
selected_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'] 
X = pd.get_dummies(titanic_df[selected_features]) 
y = titanic_df['Survived'] 

# Split and train 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) 

survival_model = RandomForestClassifier(n_estimators=100, random_state=42) 
survival_model.fit(X_train, y_train) 

# Evaluate performance 
val_accuracy = survival_model.score(X_val, y_val) 
print(f"\nValidation Accuracy: {val_accuracy * 100:.1f}%") 
print("\nKey insight: Women and children in first class had highest survival rates!")