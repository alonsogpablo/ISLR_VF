import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

x_train=pd.Series([1],[2],[3]).reshape(1,-1)
y_train=pd.Series([2],[4],[6]).reshape(1,-1)

# import model
from sklearn.linear_model import LinearRegression

# instantiate
linreg = LinearRegression()

linreg.fit(x_train,y_train)

x_test=pd.Series([4]).reshape(1,-1)

y_test=linreg.predict(x_test)

print y_test