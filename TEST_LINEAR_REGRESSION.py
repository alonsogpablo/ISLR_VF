import pandas as pd
training_data=pd.read_csv('/Users/Pablo/Desktop/Training.csv')
test_data=pd.read_csv('/Users/Pablo/Desktop/Test.csv')

feature_cols=['X']
response_col=['Y']

X_train=training_data[feature_cols]
y_train=training_data[response_col]

X_test=test_data[feature_cols]

print X_train.shape
print y_train.shape
print X_test.shape


from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

linreg.fit(X_train, y_train)

print linreg.intercept_
print linreg.coef_

y_pred = linreg.predict(X_test)

print X_test
print y_pred


