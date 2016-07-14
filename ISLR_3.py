# %load ../standard_import.txt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns

from sklearn.preprocessing import scale
import sklearn.linear_model as skl_lm
from sklearn import metrics
import statsmodels.api as sm
import statsmodels.formula.api as smf


# read CSV file directly from a URL and save the results
data = pd.read_csv('/Users/Pablo/PycharmProjects/ISLR/Advertising.csv', index_col=0)

# display the first 5 rows
print data.head()

# visualize the relationship between the features and the response using scatterplots
sns.pairplot(data, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size=7, aspect=0.7, kind='reg')
plt.show()

# create a Python list of feature names
feature_cols = ['TV', 'Radio', 'Newspaper']

# use the list to select a subset of the original DataFrame
X = data[feature_cols]

# equivalent command to do this in one line
X = data[['TV', 'Radio', 'Newspaper']]
y=data['Sales']


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# import model
from sklearn.linear_model import LinearRegression

# instantiate
linreg = LinearRegression()

# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)

# print the intercept and coefficients
print linreg.intercept_
print linreg.coef_

# pair the feature names with the coefficients
print zip(feature_cols, linreg.coef_)

# make predictions on the testing set
y_pred = linreg.predict(X_test)


print np.sqrt(metrics.mean_squared_error(y_test, y_pred))
