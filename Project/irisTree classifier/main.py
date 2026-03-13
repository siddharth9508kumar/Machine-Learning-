from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
#to fetch the data
d = load_iris()
x_tr,x_te,y_tr,y_te=train_test_split(d.data,d.target,test_size=0.2)
tree=DecisionTreeClassifier()
tree.fit(x_tr,y_tr)
y_pred=tree.predict(x_te)
accuracy=accuracy_score(y_te,y_pred)
print("Accuracy:",accuracy)
print("Predicted Class:",y_pred)