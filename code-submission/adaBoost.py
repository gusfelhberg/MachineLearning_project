import pandas
import numpy as np
import sys
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV

train_dataset = sys.argv[1]
test_dataset = sys.argv[2]
output_file = sys.argv[3]

print('Reading input data...')
train_df = pandas.read_csv(train_dataset)
test_df = pandas.read_csv(test_dataset)

X_train = train_df.drop('target',axis=1).values
y_train = train_df.target.values

# definition of the parameters that will be tested 
# to get the best combination
param_grid = { \
                'n_estimators': [100, 300, 500], \
                'learning_rate': [0.05, 0.01], \
                'loss': ['linear', 'exponential'] \
             }

print('Calculating best model...')
DTR = DecisionTreeRegressor()
ABR = AdaBoostRegressor(base_estimator = DTR)
grid_search_ABR = GridSearchCV(ABR, param_grid=param_grid)

# run the grid search with training data to select the best parameters

grid_search_ABR.fit(X_train,y_train)

#### BEST PARAMETERS
lr = grid_search_ABR.best_params_['learning_rate']
loss = grid_search_ABR.best_params_['loss']
n_estim = grid_search_ABR.best_params_['n_estimators']

print('Best parameters:')
print(grid_search_ABR.best_params_)

# build model with best parameters and predict the test data using the best parameters
regressor = AdaBoostRegressor(DecisionTreeRegressor(),
                              loss = loss,
                              n_estimators = n_estim, 
                              learning_rate = lr)
ada_model = regressor.fit(X_train,y_train)


print('Calculating predictions...')
# predict the values of the test dataset
predictions = ada_model.predict(test_df.drop('id',axis=1))

# create a pandas dataframe to handle the predictions
predictionsAndLabels = pandas.DataFrame(predictions,columns=['target'])
predictionsAndLabels['id'] = test_df.id
predictionsAndLabels = predictionsAndLabels[['id','target']]

# save the predictins in a format ready to be submitted to Kaggle
predictionsAndLabels.to_csv(output_file,index=None)


