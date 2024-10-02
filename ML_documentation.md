# ML_Docs
## Usual Imports
```
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import tensorflow_decision_forests as tfdf

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb
import optuna
```

## Data collection and analysis
Read csv data with Pandas
```
train = pd.read_csv('Train_path')
train.head()
test = pd.read_csv('Test_path')
test.head()
```
Check column null values
```
def display_missing(df):    
    for col in df.columns.tolist():          
        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))
    print('\n')
    
print("Train dataset of size", train.shape ,":")
display_missing(train)
print("Test dataset:", test.shape ,":")
display_missing(test)
```
Unique values in columns
```
def uniques(df):    
    for col in df.columns.tolist():          
        print('{} unique values: {}'.format(col, df[col].nunique()))
    print('\n')
    
print("Train dataset of size", train.shape ,":")
uniques(train)
print("Test dataset:", test.shape ,":")
uniques(test)
```
```train.loc[:,''] = train[''].fillna('')```
### Data interaction graphs
Numerical data analysis
```
for col in NUMERICAL_COLUMNS:
    plt.figure(figsize = [15, 5])
    sns.scatterplot(x=col, y='price', data=train)
    plt.title(f'Price vs {col}')
    plt.xlabel(f'{col}')
    plt.ylabel('Price')
    plt.show()
```
Categorical data analysis
```
plt.figure(figsize=(12,6))
sns.boxplot(data=train, x="brand",y = "price")
plt.show()
```

Heat map
```
temp = train.copy()
temp=temp.apply(lambda x : pd.factorize(x)[0] if x.dtype=='category'else x)
plt.figure(figsize=(16,1))
sns.heatmap(temp.corr()[6:7],annot=True,cmap='coolwarm') #change this value
plt.show()
```

## Data augmentation
Simplify categorical column values with the main point of the values - Color columns from Shiny white to white

Use regex to extract information from columns - examples:
```
cylindergroup = re.search(r'(\d+(\.\d+)?\s*)cylinder', s )
engine_cyl = int(cylindergroup.group(1)) if cylindergroup else 0

turbogroup = re.search(r'turbo', s)
turbo = True if turbogroup else False

flexgroup = re.search(r'flex fuel|flex', s)
flex_fuel = True if flexgroup else False
```
Or use lambda for the same goal
```
train['transmission_type'] = train['transmission'].apply(lambda x: 
                                                       'manual' if 'm/t' in x or 'manual' in x or  'mt' in x else 
                                                       'automatic' if 'a/t' in x or 'automatic' in x or  'at' in x else 
                                                       'CVT' if 'CVT' in x else 
                                                       'Other')

```

Split values into intervals for frequency
`train[''] = pd.qcut(train[''], 10,duplicates='drop')`

Label encoding
```
for feat in ['column']: 
    train[feat] = LabelEncoder().fit_transform(train[feat])
    test[feat] = LabelEncoder().fit_transform(test[feat])   
```
Scaler
```
scaler = StandardScaler()
for a in ['scale_columns']:
    train[a] = scaler.fit_transform(train[[a]])
    test[a] = scaler.fit_transform(test[[a]])    
```
Drop unwanted
```
train = train.drop(['column'], axis=1)
```
# Models
Basic setup
```
y = train.pop("price")
X_train, X_valid, y_train, y_valid = train_test_split(train, y, test_size=0.2, random_state=22)

useXgb = False
useGbm = True
useCatboost = False
```
## XGB Boost
```
def objective(trial):
    # Suggest hyperparameters for tuning
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': 'gbtree',
#        'n_estimators': trial.suggest_int('n_estimators', 500, 2500),
        'eta': trial.suggest_float('eta', 0.001, 0.2, log=True),  # learning rate
#         'max_depth':4,
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'lambda': trial.suggest_float('lambda', 5, 10, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10, log=True),
        'tree_method': 'hist',  
        'device':'cuda'
    }
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)

    # Train the model
    model = xgb.train(params, dtrain, evals=[(dvalid, 'validation')], num_boost_round=1500, early_stopping_rounds=25, verbose_eval=False)
    
    # Predict on the validation set
    y_pred_valid = model.predict(dvalid)
    
    # Calculate RMSE on the validation set
    rmse = mean_squared_error(y_valid, y_pred_valid, squared=False)
    
    return rmse

predictions = ''
if useXgb:
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=71)

    # Get the best hyperparameters
    best_params = study.best_params
    #print(f"Best hyperparameters: {best_params}")

    best_params['objective'] = 'reg:squarederror'
    best_params['eval_metric'] = 'rmse'
    best_params['eval_metric'] = 'rmse'
    best_params['device'] = 'cuda'
    dtrain = xgb.DMatrix(train, label=y, enable_categorical=True)
    model = xgb.train(best_params, dtrain, num_boost_round=1500)

    dtest = xgb.DMatrix(test,enable_categorical=True)  
    predictions = model.predict(dtest)
```
## LIGHT GBM
```
if useGbm:
    params = {
            'metric': 'rmse',
            'n_estimators': 1890,
            'learning_rate': 0.0049343166168420195,
            'data_sample_strategy': 'goss',
            'feature_fraction': 0.3887459059437565, 
            'lambda_l1': 7.239967197949322e-07, 
            'lambda_l2': 7.488955354504223e-06, 
            'num_leaves': 1440, 
            'max_depth': 8, 
            'colsample_bytree': 0.8390384224124089, 
            'min_child_samples': 123, 
            'min_gain_to_split': 1.491437722787296, 
            'max_bin': 246,
            'device': 'gpu',
            'verbose': -1
            
    }
    train_data = lgb.Dataset(train, label=y,params={'verbose':-1})

    tunining = False    
    if tunining:
                
        base_params_lgb={'verbose':-1,
        #                  'n_estimators':500,
                         'metric':'rmse',
                         'seed':0,
                         'data_sample_strategy': 'goss',
        #                 'n_jobs':4,#use if cpu
                         'device':'gpu',#use if gpu
                         'objective': 'regression',  
                         'max_bin': 254,
                         'feature_pre_filter':False,

                        }
        hours=2
        sampler = optuna.samplers.TPESampler()
        direction='minimize'

        def objective2(trial):

            param = {
                # 'objective': trial.suggest_categorical('objective', ['ova', 'multiclass']),
                'n_estimators': trial.suggest_int('n_estimators', 250, 2000, step=5),  
                'learning_rate': trial.suggest_float("learning_rate", 0.00005, 0.03, log=True),
        #        'data_sample_strategy': trial.suggest_categorical("data_sample_strategy", ["bagging", "goss"]),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.2, 0.99, log=True),
        #         'tree_learner': trial.suggest_categorical("tree_learner", ["serial", "feature", "data", 'voting']),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 2, 2000, step=2),
                'max_depth': trial.suggest_int('max_depth',3, 16),
        #         'subsample_for_bin': trial.suggest_int("subsample_for_bin", 40000, 600000, step=1000),
                'colsample_bytree': trial.suggest_float("colsample_bytree", 0.2, 0.99, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 150),
        #         'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-3, 10.0),
                'min_gain_to_split': trial.suggest_float('min_gain_to_split', 1e-3, 10.0),
        #        'max_bin': trial.suggest_int('max_bin', 64,254, step=2), # for GPU only till 255 is allowed!

        #         'scale_pos_weight':trial.suggest_float('scale_pos_weight', 0.0005, 10),
        #         'force_col_wise':trial.suggest_categorical("force_col_wise", [True, False]),#only used with cpu
        #         'extra_trees':True,
        #         "reg_sqrt": trial.suggest_categorical("reg_sqrt", [True, False]),
        #         "path_smooth": trial.suggest_float("path_smooth", 0, 1),
            }

        #     if param['data_sample_strategy'] == 'bagging':
        #         param['bagging_freq'] = trial.suggest_int('bagging_freq', 1, 15)
        #         param['bagging_fraction'] = trial.suggest_float('bagging_fraction', 0.4, 1.0, log=True)

        #     if param['data_sample_strategy']=='goss':
        #         param['top_rate']=trial.suggest_float('top_rate',0,0.9)
        #         param['other_rate']=trial.suggest_float('other_rate', 0,1-param['top_rate'])
        #         param['other_rate']=1-param['top_rate']

        #     if param['tree_learner'] == 'voting':
        #         param['top_k'] = trial.suggest_int('top_k', 5, 200, step=2)

            combined_params = {**param, **base_params_lgb}
            #model = lgb.LGBMRegressor(**combined_params)
            #pipe=TransformedTargetRegressor(
            #    regressor=model,
            #    transformer=StandardScaler()
            #)
            #accuracy = cross_val_score(model, X=train, y=y, cv=cv, scoring='accuracy')
            
            cv_results = lgb.cv(
                combined_params,
                train_data,
                num_boost_round=2000, 
                nfold=5,
                stratified=False,
                metrics='rmse',
            )
            rmse = cv_results['valid rmse-mean'][-1]
            print('Current trial rmse: ', rmse)
            return rmse

        study = optuna.create_study(direction=direction, pruner=optuna.pruners.MedianPruner())
        study.optimize(objective2, timeout=int(3600*hours))

        print('Number of finished trials:', len(study.trials))
        print('Best trial:', study.best_trial.params)
        params = study.best_trial.params
    
    gbm = lgb.train(params, train_data)

    predictions = gbm.predict(test)
```
## Catboost
if useCatboost:
    catboost_model = CatBoostRegressor(iterations=1000,eval_metric='RMSE', verbose=100)

    catboost_model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    predictions = catboost_model.predict(test)

# Exports
out = pd.DataFrame({
    'id': test_id,
    'class': predictions
})

out.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")
