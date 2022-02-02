# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:46:58 2022

@author: antwi
"""


import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import pickle
#from sklearn.externals import joblib


data= pd.read_csv('train.zip')

data.head()

#data.shape

data.info()

data.describe()


#sns.pairplot(data)


sns .distplot(data['paid_amount'])

sns.heatmap(data.corr(),annot=True)

data.columns

X = data.drop(['claim_id','payer_name','paid_amount','is_medicaid','is_medicare'],axis=1)
X.head()

y = data.paid_amount
y.head()


#X_enc = pd.get_dummies(X,columns= ['is_medicaid', 'is_medicare'])
#X_enc.head()



X_train, X_test, y_train, y_test=train_test_split(X, y, random_state=42, test_size=0.3)

#  RandomForestRegressor GridSearch
parameter = {
    "max_depth": [2,4,6],
    "n_estimators":[10,50,100],
    "min_samples_split":[3,5,6]
    }

grid_search = GridSearchCV(RandomForestRegressor(random_state=42),
                            param_grid = parameter)

grid_search.fit(X_train,y_train)


#printing the best parameter on the RandomForestRegressor Model
print("The best parameters are: ", grid_search.best_params_)    


# RandomForestRegressor Model and fitting
regressor = RandomForestRegressor(max_depth=6,n_estimators=100, min_samples_split=5,random_state=42)
regressor.fit(X_train,y_train)


y_pred = regressor.predict(X_test)

#joblib.dump(regressor, 'model.pkl')

#Save model
pickle.dump(regressor, open('model2.pkl', 'wb'))




