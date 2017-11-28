import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import Imputer, PolynomialFeatures, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
# from sklearn.grid_search import GridSearchCV
import pickle
import sys

def normalize(x):
    return (x - min_value) / (max_value - min_value)

train = pd.read_csv(sys.argv[1])
test = pd.read_csv(sys.argv[2])

parameters = {'n_estimators':(500, 1000, 2000),
              'max_features':[0.5, 1.0, "auto", "sqrt", "log2", None],
              'min_samples_leaf':(1, 5, 10),
              'min_samples_split':(2, 4)
             }

rfr = RandomForestRegressor( n_jobs = 8 )

rf_model = GridSearchCV(rfr, parameters)

X_train = train.drop('target',axis=1)
y_train = train.target
rf_model.fit(X_train, y_train)

print('Best Parameters:')
print(rf_model.best_params_)

# saving the best parameters dictionary into a pickle file
pickle.dump( rf_model.best_params_, open( "best_param.p", "wb" ) )

prediction = rf_model.predict(test.drop('id',axis=1))

output = pd.DataFrame(prediction)

output.columns = ['target']

output['id'] = test.id

output = output[['id', 'target']]

max_value = output.target.max()
min_value = output.target.min()

df = output.copy()

df['target'] = output.target.apply(normalize)

df.to_csv('./OutRandomForest_python_best_parameters.csv',index=False)
