

# import load_iris function from datasets module
import numpy as np
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


# save "bunch" object containing iris dataset and its attributes
iris = load_iris()

# store feature matrix in "X" (FEATURES DEL TRAINING DATASET)

X = iris.data

# store response vector in "y" (RESPONSE DEL TRAINING DATASET)
y = iris.target

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=4)

#TRAIN KNN MODEL (K=5)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

#ACCURACY SCORE
print metrics.accuracy_score(y_test, y_pred)

# instantiate the model with the best known parameters
knn = KNeighborsClassifier(n_neighbors=11)

# train the model with X and y (not X_train and y_train)
knn.fit(X, y)

# make a prediction for an out-of-sample observation
X_oos=np.array([3,5,4,2]).reshape(1,-1)

y_oos=knn.predict(X_oos)

print y_oos
