import pandas as pd
from sklearn.preprocessing import Imputer, PolynomialFeatures, StandardScaler
from sklearn.utils import shuffle


def drop_columns(dataset,columns_list):
    return dataset.drop(columns_list,axis=1)

def fill_missing_with_mean(dataset,column):
    mean_imp = Imputer(missing_values=-1, strategy='mean', axis=0)
    return mean_imp.fit_transform(dataset[[column]]).ravel()

def fill_missing_with_most_frequent(dataset,column):
    mode_imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)
    return mode_imp.fit_transform(dataset[[column]]).ravel()

def get_dummy_variables(dataset,columns_list):
    return pd.get_dummies(dataset, columns=columns_list, drop_first=True)

def create_interaction_items(dataset,columns_list):
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    interactions = pd.DataFrame(data=poly.fit_transform(dataset[columns_list]), columns=poly.get_feature_names(columns_list))
    interactions.drop(columns_list, axis=1, inplace=True)  # Remove the original columns

    return pd.concat([dataset, interactions], axis=1)

def scale_features(dataset):
    scaler = StandardScaler()
    dataset_scaled = scaler.fit_transform(dataset)
    return pd.DataFrame(dataset_scaled,columns=dataset.columns)



train = pd.read_csv('train.csv').drop('id',axis=1)
test = pd.read_csv('test.csv')


############################################
# Balancing of data (the original data has 
# much more 0 classes then 1 classes)
# Source: Kaggle data walkthrough
############################################

desired_apriori=0.10

# Get the indices per target value
idx_0 = train[train.target == 0].index
idx_1 = train[train.target == 1].index

# Get original number of records per target value
nb_0 = len(train.loc[idx_0])
nb_1 = len(train.loc[idx_1])

# Calculate the undersampling rate and resulting number of records with target=0
undersampling_rate = ((1-desired_apriori)*nb_1)/(nb_0*desired_apriori)
undersampled_nb_0 = int(undersampling_rate*nb_0)
# print('Rate to undersample records with target=0: {}'.format(undersampling_rate))
# print('Number of records with target=0 after undersampling: {}'.format(undersampled_nb_0))

# Randomly select records with target=0 to get at the desired a priori
undersampled_idx = shuffle(idx_0, random_state=37, n_samples=undersampled_nb_0)

# Construct list with remaining indices
idx_list = list(undersampled_idx) + list(idx_1)

# Return undersample data frame
train = train.loc[idx_list].reset_index(drop=True)


############################################

# Dropping the variables with too many missing values
vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']

train = drop_columns(train, vars_to_drop)
test = drop_columns(test, vars_to_drop)

# Imputing missing values with the mean or mode
train['ps_reg_03'] = fill_missing_with_mean(train,'ps_reg_03')
train['ps_car_12'] = fill_missing_with_mean(train,'ps_car_12')
train['ps_car_14'] = fill_missing_with_mean(train,'ps_car_14')
train['ps_car_11'] = fill_missing_with_most_frequent(train,'ps_car_11')

test['ps_reg_03'] = fill_missing_with_mean(test,'ps_reg_03')
test['ps_car_12'] = fill_missing_with_mean(test,'ps_car_12')
test['ps_car_14'] = fill_missing_with_mean(test,'ps_car_14')
test['ps_car_11'] = fill_missing_with_most_frequent(test,'ps_car_11')


# Creating dummy variables
categorical = ['ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat', 
               'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_04_cat', 
               'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat',
               'ps_car_09_cat', 'ps_car_10_cat', 'ps_car_11_cat' ]
train = get_dummy_variables(train,categorical)
test = get_dummy_variables(test,categorical)

# Creating interaction items
interaction_items = ['ps_reg_01','ps_reg_02','ps_reg_03','ps_car_12',
'ps_car_13','ps_car_14','ps_car_15','ps_calc_01',
'ps_calc_02','ps_calc_03']

train = create_interaction_items(train,interaction_items)
test = create_interaction_items(test,interaction_items)

# Features scaling
train = scale_features(train)

# not suposed to scale id
test_id = test.id
test = scale_features(test.drop('id',axis=1))
test['id'] = test_id


# Random permutation of train data
train = train.sample(frac=1)


# Saving files
train.to_csv('preprocessed_train.csv',index=None)
test.to_csv('preprocessed_test.csv',index=None)
