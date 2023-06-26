import os
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import gc
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_error

import lightgbm as lgb

class config:
    random_seed = 2022
    nfold = 10
    target = 'pm25_mid'

def make_corr_array(df, cols):
    output = pd.DataFrame(1 - squareform(pdist(df[cols].T, 'correlation')),
                          columns=cols, index=cols)
    return output

class AbstractBaseBlock:
    def fit(self, input_df: pd.DataFrame, y=None):
        return self.transform(input_df)
    
    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

class NumericFeatBlock(AbstractBaseBlock):
    def __init__(self, col: str):
        self.col = col
        
    def fit(self, input_df, y=None):
        pass
        
    def transform(self, input_df):
        return input_df.loc[:, self.col]

class CategoricalFeatBlock(AbstractBaseBlock):
    def __init__(self, col: str, whole_df = None, threshold=0.001, is_label=True, is_dummy=False):
        self.col = col
        self.whole_df = whole_df
        self.threshold = threshold
        self.is_label = is_label
        self.is_dummy = is_dummy
    
    def fit(self, input_df, y=None):
        if self.whole_df == None:
            df = input_df.loc[:, self.col]
        else:
            df = self.whole_df.loc[:, self.col]
        vc = df.value_counts(normalize=True).reset_index()
        vc = vc.assign(thresh=lambda d: np.where(d[self.col].values >= self.threshold, 1, 0))\
               .assign(thresh=lambda d: d['thresh'].cumsum() - d['thresh'])
        self.label_dict_ = dict(vc[['index', 'thresh']].values)
        self.label_other_ = np.max(self.label_dict_.values())
        
        return self.transform(input_df)
        
    def transform(self, input_df):
        out_df = pd.DataFrame()
        label_df = pd.DataFrame()
        label_df[f'{self.col}_label_enc'] = np.vectorize(lambda x: self.label_dict_.get(x, self.label_other_))\
                                                        (input_df[self.col].values)
        if self.is_label:
            out_df = pd.concat([out_df, label_df], axis=1)
            
        if self.is_dummy:
            label_df[f'{self.col}_label_enc'] = label_df[f'{self.col}_label_enc'].astype(object)
            out_df = pd.concat([out_df, pd.get_dummies(label_df)], axis=1)
        
        return out_df

class DateFeatureBlock(AbstractBaseBlock):
    def __init__(self, is_get_weekday=True):
        self.is_get_weekday = is_get_weekday
        
    def fit(self, input_df, y=None):
        pass
        
    def transform(self, input_df):
        out_df = pd.DataFrame()
        out_df = input_df[['year', 'month', 'day']]
        if self.is_get_weekday:
            out_df = out_df.assign(weekday=[x.dayofweek for x in input_df['timestamp'].tolist()])
        
        return out_df

class AggregateValueBlock(AbstractBaseBlock):
    def __init__(self, key_col, agg_dict, whole_df=None):
        self.key_col = key_col
        self.agg_dict = agg_dict
        self.whole_df = whole_df
        
    def fit(self, input_df, y=None):
        if self.whole_df == None:
            df = input_df
        else:
            df = self.whole_df
        agg_df = df.groupby(self.key_col).agg(self.agg_dict)
        agg_df.columns = ['_'.join(c) for c in agg_df.columns]
        self.agg_df_ = agg_df.add_prefix('_')\
                             .add_prefix('_'.join(self.key_col))\
                             .add_prefix('agg_').reset_index()
        
        return self.transform(input_df)
    
    def transform(self, input_df):
        out_df = pd.DataFrame()
        out_df = pd.merge(input_df[self.key_col], self.agg_df_, how='left', on=self.key_col)\
                   .drop(self.key_col, axis=1)\
                   .fillna(0)
        
        return out_df

def get_train_data(input_df, feat_blocks, y=None, fit_df=None):
    if fit_df is None:
        fit_df = input_df.copy()
        
    for block in feat_blocks:
        block.fit(fit_df, y)
        
    out = [block.transform(input_df) for block in feat_blocks]
    out = pd.concat(out, axis=1)
    
    return out

def get_test_data(input_df, feat_blocks):
    
    out = [block.transform(input_df) for block in feat_blocks]
    out = pd.concat(out, axis=1)
    
    return out

def cal_rmse(y_true, y_pred):
  return np.sqrt(mean_squared_error(y_true, y_pred))

def fit_lgb(x, y, cv, model_params, fit_params, fobj=None, feval=None):

    models = []
    n_records = y.shape[0]
    oof_pred = np.zeros(n_records, dtype=np.float32)
    fold = 0
    data_labels = [re.sub(r'[",\[\]{}:()]', '_', c) for c in x.columns.tolist()]

    model_params.update(deterministic = True)

    for trn_idx, val_idx in cv:

        fold += 1
        x_train, x_valid = x.iloc[trn_idx].values, x.iloc[val_idx].values
        y_train, y_valid = np.array(y.iloc[trn_idx]), np.array(y.iloc[val_idx])

        lgb_train = lgb.Dataset(x_train, y_train, feature_name=data_labels)
        lgb_valid = lgb.Dataset(x_valid, y_valid, feature_name=data_labels, reference=lgb_train)

        lgb_model = lgb.train(model_params,
                              train_set=lgb_train,
                              valid_sets=[lgb_train, lgb_valid],
                              fobj=fobj,
                              feval=feval,
                              verbose_eval=fit_params['verbose_eval'],
                              num_boost_round=fit_params['num_boost_rounds'],
                              callbacks=[lgb.early_stopping(fit_params['early_stopping_rounds'])],
                              )

        pred_valid = lgb_model.predict(x_valid, num_iteration=lgb_model.best_iteration)
        oof_pred[val_idx] = pred_valid
        models.append(lgb_model)

        print(f' - fold{fold}_RMSE - {cal_rmse(y_valid, pred_valid):4f}')

    print(f' - CV_RMSE - {cal_rmse(oof_pred, np.array(y)):4f}') 

    return oof_pred, models

def predict_test(models, df):
    out = np.array([model.predict(df) for model in models])
    out = np.where(out < 0, 0, out)
    out = np.mean(out, axis=0)
    
    return out

def plot_prediction_distribution(y_true, y_pred, y_test):
    fig, ax = plt.subplots(figsize=(18, 6))
    sns.histplot(y_test, label='Test Predict', ax=ax, color='black', stat='density')
    sns.histplot(y_pred, label='Out Of Fold', ax=ax, color='C1', stat='density', alpha=0.5)
    sns.histplot(y_true, label='True Value', ax=ax, color='blue', stat='density', alpha=0.5)
    ax.legend()
    ax.grid()    

def main():
    pd.set_option('max_rows', 400)
    pd.set_option('max_columns', 100)

    train_data = pd.read_csv('/home/kaino/comp/sony/train.csv')\
                   .assign(timestamp=lambda d: pd.to_datetime(d['year'].astype(str) + '-' + d['month'].astype(str) + '-' + d['day'].astype(str)))
    test_data = pd.read_csv('/home/kaino/comp/sony/test.csv')\
                  .assign(timestamp=lambda d: pd.to_datetime(d['year'].astype(str) + '-' + d['month'].astype(str) + '-' + d['day'].astype(str)))

    train_data['bodytemp_mid'] = 37 - ((37 - train_data['temperature_mid']) / (0.68 - (0.0014 * train_data['humidity_mid'] + (1 / (1.4 * (train_data['ws_mid'] ** 0.75) + 1.76))))) - (0.29 * train_data['temperature_mid'] * (1 - (train_data['humidity_mid'] / 100)))
    test_data['bodytemp_mid'] = 37 - ((37 - test_data['temperature_mid']) / (0.68 - (0.0014 * test_data['humidity_mid'] + (1 / (1.4 * (test_data['ws_mid'] ** 0.75) + 1.76))))) - (0.29 * test_data['temperature_mid'] * (1 - (test_data['humidity_mid'] / 100)))
    train_data['Country_Id'] = train_data['Country']
    test_data['Country_Id'] = test_data['Country']
    train_data['City_Id'] = train_data['City']
    test_data['City_Id'] = test_data['City']    

    data=pd.concat([train_data, test_data])
    le = LabelEncoder()
    for col in ["Country_Id", "City_Id"]:
        data[col] = pd.Series(le.fit_transform(data[col]))
    train_data = data[data['pm25_mid'].notna()]
    test_data = data[data['pm25_mid'].isna()]    

    corr_df = make_corr_array(train_data, [c for c in train_data.columns if re.search(r'mid|min|max|cnt|var|lat|lon|Id', c)])

    fig, ax = plt.subplots(figsize=(12, 9)) 
    sns.heatmap(corr_df, square=True, vmax=1, vmin=-1, center=0)

    corr_top10 = corr_df.loc[:, ['pm25_mid']].assign(abs_value=lambda d: np.abs(d['pm25_mid']))\
                        .sort_values('abs_value', ascending=False)\
                        .iloc[1:11, :].index.tolist()

    numeric_columns = [c for c in train_data.columns if re.search(r'max|mid|min|cnt|var|lat|lon|Id', c) and not re.search(r'pm25', c)]
    categorical_columns = ['Country']
    agg_country_value = dict([(c, [np.mean, np.max, np.min, np.std]) for c in corr_top10])

    run_blocks = [
        *[NumericFeatBlock(c) for c in [numeric_columns]],
        *[CategoricalFeatBlock(c) for c in categorical_columns],
        *[DateFeatureBlock()],
        *[AggregateValueBlock(['Country', 'month'], agg_country_value)],    
    ]

    df_train = get_train_data(train_data, run_blocks, fit_df=pd.concat([train_data, test_data], ignore_index=True))
    df_test = get_test_data(test_data, run_blocks)

    target = train_data[config.target]

    #cross validation strategy
    kf = KFold(n_splits=config.nfold, shuffle=True, random_state=config.random_seed)
    kf_cv = list(kf.split(train_data))

    lgb_model_params = {'boosting_type': 'dart',                       
                        'objective': 'rmse',

                        'learning_rate': 0.03,
                        'max_depth': -1,
                        'num_leaves': 100,
                        'min_data_in_leaf': 40,
                        'min_sum_hessian_in_leaf': 1e-2, 
                        'max_bin': 255,
                        'drop_rate': 2e-4,
                        'max_drop': 50,
                    
                        'reg_lambda': 1.0,
                        'reg_alpha': 1.,
                   
                        'colsample_bytree': 0.8,
                        'subsample': 0.8,
                        'subsample_freq': 1,
                    
                        'random_state': config.random_seed,
                        'verbose': -1,
                        'n_jobs': -1,
                     }

    lgb_fit_params = {'num_boost_rounds': 50000,
                      'early_stopping_rounds': 100,
                      'verbose_eval': 100,
                     }

    oof_valid_lgb1, lgb_models1 = fit_lgb(x=df_train, y=target, cv=kf_cv,
                                          model_params=lgb_model_params, fit_params=lgb_fit_params
                                         )                 

    pred_test = predict_test(lgb_models1, df_test)

    plot_prediction_distribution(target, oof_valid_lgb1, pred_test)

    submission = pd.read_csv('/home/kaino/comp/sony/submit_sample.csv', header=None)
    submission.iloc[:, 1] = pred_test

    submission.to_csv('/home/kaino/comp/sony/submission.csv', index=False)

if __name__ == "__main__":
    main()