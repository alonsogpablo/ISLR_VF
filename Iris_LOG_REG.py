
import numpy as np
from sklearn import metrics
# import load_iris function from datasets module
from sklearn.datasets import load_iris


# save "bunch" object containing iris dataset and its attributes
iris = load_iris()
type(iris)

#print iris.data
# print the names of the four features
#print iris.feature_names
#  print integers representing the species of each observation
#print iris.target
# check the types of the features and response
#print type(iris.data)
#print type(iris.target)

# check the shape of the features (first dimension = number of observations, second dimensions = number of features)
#print iris.data.shape

# check the shape of the response (single dimension matching the number of observations)
#print iris.target.shape

# store feature matrix in "X" (FEATURES DEL TRAINING DATASET)

X = iris.data

# store response vector in "y" (RESPONSE DEL TRAINING DATASET)
y = iris.target

# PREDICCION USANDO K NEAREST NEIGHBORS

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression ()



#AJUSTAMOS EL MODELO CON EL TRAINING DATASET
logreg.fit(X, y)

#CREAMOS UNA PREDICCION CON LOG REG

y_pred=logreg.predict(X)


from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=4)


# instantiate the model
logreg=LogisticRegression ()

# train the model with X and y (not X_train and y_train)
logreg.fit(X, y)

# make a prediction for an out-of-sample observation
X_oos=np.array([3,5,4,2]).reshape(1,-1)

y_oos=logreg.predict(X_oos)

print y_oos