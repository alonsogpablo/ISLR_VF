

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

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
#print knn

#AJUSTAMOS EL MODELO CON EL TRAINING DATASET
knn.fit(X, y)

#CREAMOS UNA PREDICCION CON KNN PARA UN NUEVO CONJUNTO DE FEATURES
X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
y_pred=knn.predict(X_new)

print y_pred

from sklearn import metrics

print metrics.accuracy_score(y, y_pred)
