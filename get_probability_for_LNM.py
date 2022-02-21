import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

import os
import pickle
import numpy as np

def load_model(name):  # for loading the pickle file
    try:
        f = open(name + '.pkl', 'rb')
    except IOError:
        return None
    else:
        return pickle.load(f)

model_dir = 'model/'

egc_te = pd.read_csv("data/samples.csv")
egc_te.astype('int32')


##################
# X vars
# Age
# Gender
# Multiple
# Location
# Gross
# Diff
# Lauren
# Depth
# Size
# LI
# VI
# PI

# Y var
# LNM
###################

def processing_dataset(df):
    X = df[["Age", "Gender", "Multiple", "Location", "Gross", "Diff",
            "Lauren", "Depth", "Size", "LI", "VI", "PI"]]

    X_con = df[["Age", "Size"]].copy()
    X_con["Age"] = (X_con["Age"].values - 57.88975996902826) / 11.108860394601379 # Standardization. The values were from training set.
    X_con["Size"] = (X_con["Size"] - 26.93815331010453) / 17.73154099949414

    X_new = pd.concat([
        pd.get_dummies(X['Gender'], prefix='Gender'),
        pd.get_dummies(X['Multiple'], prefix='Multiple'),
        pd.get_dummies(X['Location'], prefix='Location'),
        pd.get_dummies(X['Gross'], prefix='Gross'),
        pd.get_dummies(X['Diff'], prefix='Diff'),
        pd.get_dummies(X['Lauren'], prefix='Lauren'),
        pd.get_dummies(X['Depth'], prefix='Depth'),
        pd.get_dummies(X['LI'], prefix='LI'),
        pd.get_dummies(X['VI'], prefix='VI'),
        pd.get_dummies(X['PI'], prefix='PI'),
        pd.DataFrame(X_con, columns=['Age', 'Size']),
    ],
        axis=1)
    Y = df["LNM"]
    return X_new, Y


x_te, y_te = processing_dataset(egc_te)

for i in range(3):
    model_names = [
        'Logistic',
        'RandomForest',
        'SVM'
    ]
    model = load_model(os.path.join(model_dir, model_names[i]+'_1'))

    y_pr = model.predict_proba(x_te)
    '''
        ** expected threshold for LNM is 0.77 according to the rate of LNM in the training set.
        ** If the predicted probability for class '1' is over 0.77, then the model predict the patient as 'LNM'
    '''
    #fpr, tpr, thresholds = metrics.roc_curve(y_te, y_pr[:, 1])

    train_pred = pd.concat([egc_te.reset_index(drop=True), pd.DataFrame(y_pr[:, 1], columns=['PRED'])], axis=1)
    train_pred.to_csv(os.path.join(model_dir, 'RSV_' +  model_names[i] + '_pred.csv'), index=False)
