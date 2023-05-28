import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#import tensorflow as tf

# загрузка данных
path_X_train = 'X_train.parquet'
path_X_test = 'X_test.parquet'
path_y_train = 'y_train.parquet'

df_X_train = pd.read_parquet(path_X_train, engine='pyarrow')
df_X_test = pd.read_parquet(path_X_test, engine='pyarrow')
df_y_train = pd.read_parquet(path_y_train, engine='pyarrow')

NUM_ATRIBUTES = len(df_X_train.keys())

# заполняем пропуски
df_X_test.fillna(method='ffill', inplace=True)
df_X_test.fillna(method='bfill', inplace=True)
df_X_train.fillna(method='ffill', inplace=True)
df_X_train.fillna(method='bfill', inplace=True)

# фильтрация скользящим усреднением
NUM_PERIODS = 10
LIST_OF_KEYS = list(df_X_train.keys())
for i, key in enumerate(LIST_OF_KEYS):
    df_X_train[i] = df_X_train[key].rolling(NUM_PERIODS).mean()
    df_X_train.drop(labels=[key], inplace=True, axis=1)
for i, key in enumerate(LIST_OF_KEYS):
    df_X_test[i] = df_X_test[key].rolling(NUM_PERIODS).mean()
    df_X_test.drop(labels=[key], inplace=True, axis=1)

# заполняем пропуски после фильтрации
for i, key in enumerate(df_X_test.keys()):
    df_X_test.iloc[:NUM_PERIODS - 1, i] = df_X_test.iloc[NUM_PERIODS - 1, i]
    df_X_train.iloc[:NUM_PERIODS - 1, i] = df_X_train.iloc[NUM_PERIODS - 1, i]

# стандартезация по параметрам X_train
normalized_X_test=(df_X_test-df_X_train.mean())/df_X_train.std()
normalized_X_test = normalized_X_test.astype('float16')
normalized_X_test.to_csv(path_or_buf='normalized_X_test.csv')
                         
normalized_X_train=(df_X_train-df_X_train.mean())/df_X_train.std()
normalized_X_train = normalized_X_train.astype('float16')
normalized_X_train.to_csv(path_or_buf='normalized_X_train.csv')

df_y_train = df_y_train.astype(int)
y_train_for_NN = (df_y_train == 2) * 1
y_train_for_NN = y_train_for_NN.astype('int8')
y_train_for_NN.to_csv(path_or_buf='y_train_for_NN.csv')
