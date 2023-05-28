import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pickle

path_X_test = 'normalized_X_test.csv'
normalized_X_test = pd.read_csv(path_X_test)
index = normalized_X_test['DT']
del normalized_X_test
dic = {}
for i in range(175):
    with open(f'\\predictions\\{i}.pickle', 'rb') as f:
        data_new = pickle.load(f)
    dic.update(data_new)
df = pd.DataFrame(data=dic, index=index)
df.to_csv(path_or_buf='my_submission_3.csv')
