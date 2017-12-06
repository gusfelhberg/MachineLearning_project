import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle
import sys

def normalize(x):
    return (x - min_value) / (max_value - min_value)

train = pd.read_csv(sys.argv[1])
test = pd.read_csv(sys.argv[2])
output = sys.argv[3]

param_grid = {'n_estimators':(500, 1000, 2000),
              'max_features':[0.5, 1.0, "auto", "sqrt", "log2", None],
              'min_samples_leaf':(1, 5, 10),
              'min_samples_split':(2, 4)
             }

rfr = RandomForestRegressor()

rf_model = GridSearchCV(rfr, param_grid)

X_train = train.drop('target',axis=1)
y_train = train.target

# run the grid search with training data to select the best parameters
rf_model.fit(X_train, y_train)

print('Best Parameters:')
print(rf_model.best_params_)

# predict the test data using the best parameters
prediction = rf_model.predict(test.drop('id',axis=1))

# create a pandas dataframe to handle the predictions
output = pd.DataFrame(prediction)
output.columns = ['target']
output['id'] = test.id
output = output[['id', 'target']]

# normalize the results to the range 0-1
max_value = output.target.max()
min_value = output.target.min()
df = output.copy()
df['target'] = output.target.apply(normalize)

# save the predictins in a format ready to be submitted to Kaggle
df.to_csv(output,index=False)
