import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from itertools import repeat
from collections import Counter
from sklearn import preprocessing


# load data
def load_data():
    x_vals = pd.read_csv('data/train_call.txt', delimiter='\t')
    y_vals = pd.read_csv('data/train_clinical.txt', delimiter='\t')
    return x_vals.transpose(), y_vals


# Define base model
def init_model():
    classifier = xgb.XGBClassifier(
      booster='gbtree',
      learning_rate=.003,
      max_delta_step=0,
      max_depth=16,
      min_child_weight=1,
      n_estimators=256,
      objective='binary:logistic',
      random_state=0,
      eval_metric=accuracy_score,
    )
    return classifier


""" SHAP-BASED FEATURE SELECTION"""
# it would be simple enough to extract most relevant features, but because of a
# naive approach where we select the features after the initial modeling, it
# makes a difference which train-validation-split we take. Therefore we do an
# initial cross-validation step to identify which features most commonly appear.
# This is done by simply taking the most frequently+highly ranked features

def get_shap_features(X, y, shap_cv=25):
    # we will add multiples of features to this list, to also take into account feature importance
    scores = [0] * X.shape[1]

    # start CV loop
    for i in range(shap_cv):
        gbc_shap = init_model()
        # train test split and model-fitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        gbc_shap.fit(X_train, y_train)

        # use SHAP explainer
        explainer = shap.TreeExplainer(gbc_shap)
        shap_values = explainer.shap_values(X_test)

        # get k most important features
        feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
        k_best_features = feature_order[-50:]
        for k, feat in enumerate(k_best_features):
            # add importance of feature (reversed rank) to scores list
            scores[feat] += k+1

    return scores, [1 for number in scores]


""" PRE-PROCESSING """
# get data and drop unusable columns
X, y = load_data()
drop_rows = ['Chromosome', 'Start', 'End', 'Nclone']
X = X.drop(index=drop_rows)
y = y.set_index('Sample')

# encode labels (HER2+ = 0, HR+ = 1, Triple Neg = 2)
target_names = ['HER2+', 'HR+', 'Triple Neg']
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(y)
y = pd.DataFrame(label_encoder.transform(y))

# Define preprocessing
scaler = preprocessing.MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X))
# These are the experiments, i.e. the selectors that are compared
selectors = [SelectKBest(get_shap_features, k=250),
             SelectKBest(mutual_info_classif, k=250),
             SelectKBest(chi2, k=250)]

# Now we remove the "0" labeled samples from the train set
X = X[y[0] != 0]
y = y[y[0] != 0]
y = y - 1


for i, selector in enumerate(['SHAP', 'MutualInfo', 'Chi2']):
    select = selectors[i]
    z = select.fit_transform(X, y)
    filter = select.get_support()
    features = np.array(X.columns)
    print(f'\n\n\nselected features for {selector}:\n\n', features[filter])