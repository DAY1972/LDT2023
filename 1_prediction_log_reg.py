import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pickle

# загрузка данных
path_X_train = 'normalized_X_train.csv'
path_X_test = 'normalized_X_test.csv'
path_y_train = 'y_train_for_NN.csv'

col_X_train = pd.read_csv(path_X_train, nrows=2)
# ограничение по ОЗУ skiprows=4000000 
normalized_X_train = pd.read_csv(path_X_train, skiprows=4000000,
                                 names=col_X_train.keys())
normalized_X_train.set_index('DT', inplace=True)
normalized_X_train = normalized_X_train.astype('float16')

normalized_X_test = pd.read_csv(path_X_test)
normalized_X_test.set_index('DT', inplace=True)
normalized_X_test = normalized_X_test.astype('float16')

col_y_train = pd.read_csv(path_y_train, nrows=2)
# ограничение по ОЗУ skiprows=4000000
y_train_for_NN = pd.read_csv(path_y_train, skiprows=4000000,
                             names=col_y_train.keys())
y_train_for_NN.set_index('DT', inplace=True)
y_train_for_NN = y_train_for_NN.astype('int8')

for i, key in enumerate(y_train_for_NN.keys()):
    #print(i)
    prediction = {}
    if (y_train_for_NN[key].sum() > 10) and (y_train_for_NN[key].sum() < y_train_for_NN.shape[0]):
        X_tr, X_val, y_tr, y_val = train_test_split(normalized_X_train,
                                                    y_train_for_NN[key],
                                                    test_size=0.2, random_state=100,
                                                    stratify=y_train_for_NN[key])
        clf = LogisticRegression(random_state=0, max_iter=100).fit(X_tr, y_tr)
        pred_val = clf.predict_proba(X_val)
        levels = np.linspace(0.05, 0.9, num=18)
        metric = np.zeros_like(levels)
        for j, level in enumerate(levels):
            y_pred = (pred_val[:, 1] >= level) * 1
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            metric[j] = tp / (tp + fn + fp)
        level = levels[np.argmax(metric)]
        pred_test = clf.predict_proba(normalized_X_test)
    else:
        pred_test = np.zeros(shape=(normalized_X_test.shape[0], 2), dtype=int)
        if y_train_for_NN[key].sum() >= y_train_for_NN.shape[0] - 1:
            pred_test = pred_test + 1
        level = 1
    prediction[key] = (pred_test[:, 1] >= level) * 2
    with open(f'\predictions\\{i}.pickle', 'wb') as f:
        pickle.dump(prediction, f)
