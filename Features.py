import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error
pd.options.display.precision = 15

import lightgbm as lgb
import xgboost as xgb
import time
import datetime
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import gc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from scipy import stats
from sklearn.kernel_ridge import KernelRidge

from tsfresh import extract_features
from tsfresh import utilities
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import feature_extraction


n_jobs=4
fcp = {
    'ad': {'absolute_sum_of_changes': None, 'cid_ce': [{'normalize' : 1}], 'maximum' : None, 'minimum' : None,
        'mean' : None, 'median' : None, 'variance' : None, 'standard_deviation' : None, 'kurtosis' : None, 'skewness' : None,
        'mean_change' : None, 'mean_second_derivative_central' : None, 'number_peaks' : [{'n' : 3}],
        'ratio_beyond_r_sigma' : [{'r' : 0.5}, {'r' : 1}, {'r' : 1.5}, {'r' : 2}, {'r' : 2.5}, {'r' : 3}]},
    'ad_n': {'abs_energy': None, 'quantile': [{'q' : 0.99},{'q' : 0.98},{'q' : 0.97},{'q' : 0.96},{'q' : 0.95},{'q' : 0.94},
         {'q' : 0.93},{'q' : 0.92},{'q' : 0.91},{'q' : 0.90},{'q' : 0.85},{'q' : 0.80},{'q' : 0.20},{'q' : 0.15},{'q' : 0.10},
         {'q' : 0.09},{'q' : 0.08},{'q' : 0.07},{'q' : 0.06},{'q' : 0.05},{'q' : 0.04},{'q' : 0.03},{'q' : 0.02},{'q' : 0.01}],
        'number_crossing_m' : [{'m' : 5},{'m' : 10},{'m' : 20}],
        'fft_aggregated' : [{'aggtype' : 'centroid'},{'aggtype' : 'variance'},{'aggtype' : 'skew'},{'aggtype' : 'kurtosis'}]},
    'ad_ene': {'abs_energy': None},
}

def ftrs_5(df5):
    ft5=pd.DataFrame(dtype=np.float64)
    ft5.loc[0, 'ene_0']=df5['acd_n__abs_energy'].values[4]*5 / \
                                        df5['acd_n__abs_energy'].values[0:5].sum()
    ft5.loc[0, 'ene_1']=df5['acd_n__abs_energy'].values[3:5].sum()*2.5 / \
                                        df5['acd_n__abs_energy'].values[0:5].sum()
    ft5.loc[0, 'ene_2']=np.prod(df5['acd_n__abs_energy'].values[3:5])**0.5 / \
                                        np.prod(df5['acd_n__abs_energy'].values[0:5])**0.2
    ft5.loc[0, 'ene_3']=df5['acd_n__abs_energy'].values[2:5].sum()*(5/3) / \
                                        df5['acd_n__abs_energy'].values[0:5].sum()
    ft5.loc[0, 'ene_4']=np.prod(df5['acd_n__abs_energy'].values[2:5])**(1/3) / \
                                        np.prod(df5['acd_n__abs_energy'].values[0:5])**0.2
    ft5.loc[0, 'ene_5']=(df5['acd_n__abs_energy'].values[4]+df5['acd_n__abs_energy'].values[0])*0.5 / \
                                        df5['acd_n__abs_energy'].values[2]
    ft5.loc[0, 'ene_6']=(df5['acd_n__abs_energy'].values[4]*df5['acd_n__abs_energy'].values[0])**0.5 / \
                                        df5['acd_n__abs_energy'].values[2]
    ft5.loc[0, 'ene_7']=(df5['acd_n__abs_energy'].values[4]+df5['acd_n__abs_energy'].values[0])*1.5 / \
                                        df5['acd_n__abs_energy'].values[1:4].sum()
    ft5.loc[0, 'ene_8']=(df5['acd_n__abs_energy'].values[4]*df5['acd_n__abs_energy'].values[0])**0.5 / \
                                        np.prod(df5['acd_n__abs_energy'].values[1:4])**(1/3)
    ft5.loc[0, 'ene_9']=df5['acd_n__abs_energy'].values[3:5].sum() / \
                                        df5['acd_n__abs_energy'].values[0:2].sum()
    ft5.loc[0, 'ene_10']=np.prod(df5['acd_n__abs_energy'].values[3:5])**0.5 / \
                                        np.prod(df5['acd_n__abs_energy'].values[0:2])**0.5
    ft5.loc[0, 'ene_11']=(df5['acd_n__abs_energy'].values[4]+df5['acd_n__abs_energy'].values[0]) / \
                                        (df5['acd_n__abs_energy'].values[3]+df5['acd_n__abs_energy'].values[1])
    ft5.loc[0, 'ene_12']=(df5['acd_n__abs_energy'].values[4] * df5['acd_n__abs_energy'].values[0])**0.5 / \
                                        (df5['acd_n__abs_energy'].values[3] * df5['acd_n__abs_energy'].values[1])**0.5
    ft5.loc[0, 'ene_13']=(df5['acd_n__abs_energy'].values[4]+df5['acd_n__abs_energy'].values[1]) / \
                                        (df5['acd_n__abs_energy'].values[3]+df5['acd_n__abs_energy'].values[0])
    ft5.loc[0, 'ene_14']=(df5['acd_n__abs_energy'].values[4] * df5['acd_n__abs_energy'].values[1])**0.5 / \
                                        (df5['acd_n__abs_energy'].values[3] * df5['acd_n__abs_energy'].values[0])**0.5
    ft5.loc[0, 'ene_15']=(df5['acd_n__abs_energy'].values[4]+df5['acd_n__abs_energy'].values[2]) / \
                                        (df5['acd_n__abs_energy'].values[3]+df5['acd_n__abs_energy'].values[1])
    ft5.loc[0, 'ene_16']=(df5['acd_n__abs_energy'].values[4] * df5['acd_n__abs_energy'].values[2])**0.5 / \
                                        (df5['acd_n__abs_energy'].values[3] * df5['acd_n__abs_energy'].values[1])**0.5
    ft5.loc[0, 'ene_17']=df5['acd_n__abs_energy'].values[4] / df5['acd_n__abs_energy'].values[0]
    ft5.loc[0, 'ene_18']=df5['acd_n__abs_energy'].values[4]*2 / df5['acd_n__abs_energy'].values[0:2].sum()
    ft5.loc[0, 'ene_19']=df5['acd_n__abs_energy'].values[4] / \
                                        np.prod(df5['acd_n__abs_energy'].values[0:2])**0.5
    ft5.loc[0, 'ene_20']=df5['acd_n__abs_energy'].values[4]*3 / df5['acd_n__abs_energy'].values[0:3].sum()
    ft5.loc[0, 'ene_21']=df5['acd_n__abs_energy'].values[4] / \
                                        np.prod(df5['acd_n__abs_energy'].values[0:3])**(1/3)
    ft5.loc[0, 'ene_22']=df5['acd_n__abs_energy'].values[4]*4 / df5['acd_n__abs_energy'].values[0:4].sum()
    ft5.loc[0, 'ene_23']=df5['acd_n__abs_energy'].values[4] / \
                                        np.prod(df5['acd_n__abs_energy'].values[0:4])**(1/4)
    ft5.loc[0, 'ene_24']=df5['acd_n__abs_energy'].values[3:5].sum()*2 / \
                                        df5['acd_n__abs_energy'].values[0:4].sum()
    ft5.loc[0, 'ene_25']=np.prod(df5['acd_n__abs_energy'].values[3:5])**0.5 / \
                                        np.prod(df5['acd_n__abs_energy'].values[0:4])**0.2
    ft5.loc[0, 'ene_26']=df5['acd_n__abs_energy'].values[2:5].sum() / \
                                        df5['acd_n__abs_energy'].values[0:3].sum()
    ft5.loc[0, 'ene_27']=np.prod(df5['acd_n__abs_energy'].values[2:5])**(1/3) / \
                                        np.prod(df5['acd_n__abs_energy'].values[0:3])**(1/3)
    ft5.loc[0, 'ene_28']=df5['acd_n__abs_energy'].values[2:5].sum()*4 / \
                                        df5['acd_n__abs_energy'].values[0:4].sum()*3
    ft5.loc[0, 'ene_29']=np.prod(df5['acd_n__abs_energy'].values[2:5])**(1/3) / \
                                        np.prod(df5['acd_n__abs_energy'].values[0:4])**(1/4)
    ft5.loc[0, 'ene_30']=df5['acd_n__abs_energy'].values[1:5].sum() / \
                                        df5['acd_n__abs_energy'].values[0:4].sum()
    ft5.loc[0, 'ene_31']=np.prod(df5['acd_n__abs_energy'].values[1:5])**(1/4) / \
                                        np.prod(df5['acd_n__abs_energy'].values[0:4])**(1/4)
    return ft5


def featurize1(df, param):
    ft = pd.DataFrame(dtype=np.float64)
    ft_2 = pd.DataFrame(dtype=np.float64)
    ft_5 = pd.DataFrame(dtype=np.float64)
    
    df = df.reset_index(drop=True)
    df['acd_n']=df['acd']-df['acd'].mean()
    df['id_1']=0
    df['time_1']=df.index
    df['id_2']=df.index//(df.shape[0]//2)
    df['time_2']=df.index%(df.shape[0]//2)
    df['id_5']=df.index//(df.shape[0]//5)
    df['time_5']=df.index%(df.shape[0]//5)

    ft = extract_features(df, column_id='id_1', column_sort='time_1', column_value='acd', 
                                               default_fc_parameters=param['ad'], n_jobs=n_jobs, disable_progressbar=True)
    ft = pd.concat([ft, extract_features(df, column_id='id_1', column_sort='time_1', column_value='acd_n', 
                                               default_fc_parameters=param['ad_n'], n_jobs=n_jobs, disable_progressbar=True)], axis=1)
    
    ft_2 = extract_features(df, column_id='id_2', column_sort='time_2', column_value='acd', 
                                               default_fc_parameters=param['ad'], n_jobs=n_jobs, disable_progressbar=True)
    ft_2 = pd.concat([ft_2, extract_features(df, column_id='id_2', column_sort='time_2', column_value='acd_n', 
                                               default_fc_parameters=param['ad_n'], n_jobs=n_jobs, disable_progressbar=True)], axis=1)
    ft['acd_n__fft_aggregated__aggtype_"kurtosis"'] = ft_2['acd_n__fft_aggregated__aggtype_"kurtosis"'].sum()/2
    ft['acd_n__fft_aggregated__aggtype_"skew"'] = ft_2['acd_n__fft_aggregated__aggtype_"skew"'].sum()/2
    for i in ft.columns:
        ft[i+'_r2']=ft_2[i].iloc[1]/(ft_2[i].iloc[0]+0.0001)
        
    ft_5 = extract_features(df, column_id='id_5', column_sort='time_5', column_value='acd_n',
                                               default_fc_parameters=param['ad_ene'], n_jobs=n_jobs, disable_progressbar=True)
    ft = pd.concat([ft, ftrs_5(ft_5)], axis=1)
    return ft


def featurize0(df, tr_f):
    fcp['ad_n']['number_crossing_m']=[{'m' : 5},{'m' : 10},{'m' : 20}]
    ft_0 =featurize1(df, fcp)
    
    fcp['ad_n']['number_crossing_m']=[{'m': 2.5}, {'m': 5}, {'m': 10}]
    df_10_mean=pd.DataFrame(df['acd'].rolling(11).mean().iloc[10:].values, columns=['acd'])
    ft_10_mean =featurize1(df_10_mean, fcp)
    ft_10_mean.columns=ft_10_mean.columns+'_r10m'
    fcp['ad_n']['number_crossing_m']=[{'m': 4}, {'m': 8}, {'m': 16}]
    df_10_std=pd.DataFrame(df['acd'].rolling(11).std().iloc[10:].values, columns=['acd'])
    ft_10_std =featurize1(df_10_std, fcp)
    ft_10_std.columns=ft_10_std.columns+'_r10s'
    
    fcp['ad_n']['number_crossing_m']=[{'m': 0.4}, {'m': 0.8}, {'m': 1.6}]
    df_100_mean=pd.DataFrame(df['acd'].rolling(101).mean().iloc[100:].values, columns=['acd'])
    ft_100_mean =featurize1(df_100_mean, fcp)
    ft_100_mean.columns=ft_100_mean.columns+'_r100m'
    fcp['ad_n']['number_crossing_m']=[{'m': 3}, {'m': 6}, {'m': 12}]
    df_100_std=pd.DataFrame(df['acd'].rolling(101).std().iloc[100:].values, columns=['acd'])
    ft_100_std =featurize1(df_100_std, fcp)
    ft_100_std.columns=ft_100_std.columns+'_r100s'
    
    fcp['ad_n']['number_crossing_m']=[{'m': 0.125}, {'m': 0.25}, {'m': 0.5}]
    df_1000_mean=pd.DataFrame(df['acd'].rolling(1001).mean().iloc[1000:].values, columns=['acd'])
    ft_1000_mean =featurize1(df_1000_mean, fcp)
    ft_1000_mean.columns=ft_1000_mean.columns+'_r1000m'
    fcp['ad_n']['number_crossing_m']=[{'m': 3}, {'m': 6}, {'m': 12}]
    df_1000_std=pd.DataFrame(df['acd'].rolling(1001).std().iloc[1000:].values, columns=['acd'])
    ft_1000_std =featurize1(df_1000_std, fcp)
    ft_1000_std.columns=ft_1000_std.columns+'_r1000s'
    
    ft=pd.concat([ft_0, ft_10_mean, ft_10_std, ft_100_mean, ft_100_std, ft_1000_mean, ft_1000_std], axis=1)
    
    if tr_f==True:
        return ft, df['time_to_failure'].iloc[-1]
    else:
        return ft

fet = pd.DataFrame(dtype=np.float64)
y = pd.DataFrame(dtype=np.float64)
for i in tqdm_notebook(range(17)):
    train = pd.read_csv('split/train_{:02}.csv'.format(i), dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
    train.columns=['acd', 'time_to_failure']
    
    for j in range(train.shape[0]//37500-2):
        if j == train.shape[0]//37500-2:
            df=train.iloc[-150000:, :]
        else:
            df = train.iloc[j*37500:j*37500+150000, :]
        tr_1, y_1=featurize0(df,True)
        y =pd.concat([y, pd.DataFrame([y_1], columns=['time_to_failure'], dtype=np.float64)], axis=0)
        fet=pd.concat([fet, tr_1], axis=0)


y.index=[i for i in range(y.shape[0])]
fet.index=[i for i in range(fet.shape[0])]

y_tr = pd.DataFrame(y, dtype=np.float64, columns=['time_to_failure'])
X_tr = fet
submission = pd.read_csv('input/sample_submission.csv', index_col='seg_id')
X_test = pd.DataFrame()


for i in tqdm_notebook(range(submission.shape[0])):
    seg = pd.read_csv('input/test/' + submission.index[i] + '.csv')
    seg.columns=['acd']
    X_test = pd.concat([X_test, featurize0(seg,False)], axis=0)
    
X_test.index=[i for i in submission.index]

y_tr.to_csv('y_tr_06.csv')
X_tr.to_csv('X_tr_06.csv')
X_test.to_csv('X_test_06.csv')