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


# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train, y_train)

# predict the response for new observations
y_test=logreg.predict(X_test)


print y_test