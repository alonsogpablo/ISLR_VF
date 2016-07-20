import  pandas as pd

import pandas as pd
training_data=pd.read_csv('/Users/Pablo/Desktop/TrainKNN.csv')
test_data=pd.read_csv('/Users/Pablo/Desktop/TestKNN.csv')

feature_cols=['DENSIDAD']
response_col=['TIPO']

X_train=training_data[feature_cols]
y_train=training_data[response_col]

X_test=test_data[feature_cols]

print X_train.shape
print y_train.shape
print X_test.shape


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)  #ELEGIMOS K=5
knn.fit(X_train, y_train)

y_test=knn.predict(X_test)

print y_test