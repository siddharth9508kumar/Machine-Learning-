# Import necessary modules 
from sklearn.datasets import load_digits 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
import matplotlib.pyplot as plt 

# Load the digits dataset 
digit_data = load_digits()

# if u want u can also see the dighits
#to see the digit remove the comment

'''# Display sample digits 
fig, axes = plt.subplots(2, 4, figsize=(10, 5)) 
for idx, ax in enumerate(axes.flatten()): 
    ax.imshow(digit_data.images[idx], cmap='gray') 
    ax.set_title(f"Digit: {digit_data.target[idx]}") 
    ax.axis('off') 
plt.tight_layout() 
plt.show() '''

# Prepare and split data 
X_train, X_test, y_train, y_test = train_test_split( 
    digit_data.data, digit_data.target, test_size=0.2, random_state=42) 

# Train Random Forest classifier 
rf_model = RandomForestClassifier(n_estimators=100, random_state=42) 
rf_model.fit(X_train, y_train) 

# Calculate and display accuracy 
accuracy = rf_model.score(X_test, y_test) 
print(f"\nDigit Recognition Accuracy: {accuracy * 100:.1f}%") 