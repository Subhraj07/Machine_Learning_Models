import pandas as pd
import numpy as np

advt = pd.read_csv( "MLModels/datasets/Advertising.csv" )
advt.head(6)
advt.info()
#remove first column advt = advt.iloc[:,1:6]
advt = advt[["TV", "radio", "newspaper", "sales"]]
X = advt.iloc[:, [0,1,2]].values
y = advt.iloc[:, 3].values
#building the model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)
len( X_train )
len( X_test )

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit( X_train, y_train )
regressor.intercept_
list( zip( ["TV", "Radio", "Newspaper"], list( regressor.coef_ ) ) )

#making predictions

y_pred = regressor.predict( X_test )

#comparing predictions with actuals
test_pred_df = pd.DataFrame( { 'actual': y_test,
                            'predicted': np.round( y_pred, 2),
                            'residuals': y_test - y_pred } )
test_pred_df[0:10]

#measuring model
from sklearn import metrics
rmse = np.sqrt( metrics.mean_squared_error( y_test, y_pred ) )
round( rmse, 2 )

y_test.mean()
y_pred.mean()
#evaluating the model accuracy
metrics.r2_score( y_test, y_pred )

#understanding residuals
import matplotlib.pyplot as plt
import seaborn as sn
%matplotlib 

residuals = y_test - y_pred

sn.jointplot(advt.sales, residuals, size = 6)

sn.distplot( residuals )

#Creating a new features and rebuilding the model - synergy effect 
#Perhaps spending $50,000 on television advertising and $50,000 on radio advertising results in more sales 
#than allocating $100,000 to either television or radio individually. 
#In marketing, this is known as a synergy effect, while in statistics it is called an interaction effect.

#let's create an interaction variable for TV and Radio Spending
advt['tv_radio'] = advt.TV * advt.radio


#advt = advt.iloc[:,1:6]
#X = advt.iloc[:, [0,1,2,4]].values
#y = advt.iloc[:, 3].values
#building the model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
  advt[["TV", "radio", "newspaper", "tv_radio"]],
  advt.sales,
  test_size=0.3,
  random_state = 42 )


len( X_train )
len( X_test )

linreg = LinearRegression()
linreg.fit( X_train, y_train )

y_pred = linreg.predict( X_test )

metrics.r2_score(y_test, y_pred)

residuals = y_test - y_pred

sn.jointplot( advt.sales, residuals, size = 6 )
sn.distplot( residuals )
#####K-Fold Cross Validation
from sklearn.model_selection import cross_val_score

linreg = LinearRegression()
cv_scores = cross_val_score( linreg, X_train, y_train, scoring = 'r2', cv = 10 )
cv_scores

print( "Average r2 score: ", np.round( np.mean( cv_scores ), 2 ) )
print( "Standard deviation in r2 score: ", np.round( np.std( cv_scores ), 2) )

#####Feature Ranking based on importance
#Here is what f_regression does, on input matrix XX and array yy. 
#For every feature it computes the correlation with y: Then it computes the F-statistic
#Fi=ρ2i/(1−ρ2i)
#These F-values are then returned, together with the associated p-values. 
#So the result is a tuple (F-values, p-values). 
#Then SelectKBest takes the first component of this tuple (these will be the scores), 
#sorts it, and picks the first k features of X with the highest scores.

from sklearn.feature_selection import SelectKBest, f_regression
model = SelectKBest( score_func=f_regression, k=4 )

results = model.fit( X_train, y_train )

results.scores_

results.pvalues_

['%.3f' % p for p in results.pvalues_]

#As p - values are less than 5% - the variables are siginificant in the regression equation. 
#Also higher the F value, higher is the importance. So, as per this test, we can rank the features as below.
#1. TV*Radio, 2. TV, 3. Radio, 4. Newspaper
#[  TV*Radio: 185.64138393,    Radio: 88.09887658,     Newspaper: 8.83792204,  TV*Radio: 1681.74689385]

##########Building and Exporting the model 
linreg = LinearRegression()
linreg.fit( X_train, y_train )

import pickle

from sklearn.externals import joblib
joblib.dump(linreg, 'lin_model.pkl', compress=9)


#########Importing and applying the model for prediction
model_clone = joblib.load('lin_model.pkl')

model_clone.intercept_


model_clone.coef_

pred_y = model_clone.predict( X_test )                   

############################################################33





#http://www.awesomestats.in/python-regression-advertisement/