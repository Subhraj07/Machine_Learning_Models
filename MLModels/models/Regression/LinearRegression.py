import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

train = pd.read_csv('./dataset/Train_UWu5bXk.csv')
test = pd.read_csv('./dataset/Test_u94Q5KV.csv')

# lreg = LinearRegression()
#
# X = train.loc[:, ['Outlet_Establishment_Year', 'Item_MRP']]
# x_train, x_cv, y_train, y_cv = train_test_split(X, train.Item_Outlet_Sales)
# lreg.fit(x_train, y_train)
# pred = lreg.predict(x_cv)
# mse = np.mean((pred - y_cv) ** 2)
# coeff = pd.DataFrame(x_train.columns)
# coeff['Coefficient Estimate'] = pd.Series(lreg.coef_)
# # R-Square
# lreg.score(x_cv, y_cv)

# Linear regression with more variables
lreg = LinearRegression()

X = train.loc[:, ['Outlet_Establishment_Year','Item_MRP','Item_Weight']]
x_train, x_cv, y_train, y_cv = train_test_split(X, train.Item_Outlet_Sales)
lreg.fit(x_train, y_train)
pred = lreg.predict(x_cv)
mse = np.mean((pred - y_cv) ** 2)
coeff = pd.DataFrame(x_train.columns)
coeff['Coefficient Estimate'] = pd.Series(lreg.coef_)
# R-Square
lreg.score(x_cv, y_cv)

print("Completed")
