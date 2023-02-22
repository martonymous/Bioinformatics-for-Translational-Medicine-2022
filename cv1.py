## Imports
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn import metrics
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.metrics import f1_score, accuracy_score, classification_report
import statistics


## load data
def load_data():
    x_vals = pd.read_csv('data/train_call.txt', delimiter='\t')
    y_vals = pd.read_csv('data/train_clinical.txt', delimiter='\t')
    return x_vals.transpose(), y_vals


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

# Define base model
def init_model():
    classifier = xgb.XGBClassifier(
      booster='gbtree',
      learning_rate=.003,
      max_delta_step=0,
      max_depth=16,
      min_child_weight=1,
      n_estimators=256,
      objective='multi:softmax',
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
    # we will add reversed rank of features to this list, to also take into account feature importance
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
        feature_order = np.argsort(np.sum(np.sum(np.abs(shap_values), axis=0), axis=0))
        k_best_features = feature_order[-50:]
        for k, feat in enumerate(k_best_features):
            # add importance of feature (reversed rank) to scores list
            scores[feat] += k + 1

    return scores, [1 for number in scores]


""" NESTED CROSS VALIDATION """
# Define nested cross-validation parameters (NCV)
outer_cv = 8

# Define model parameter search space
mut_search_space = [{'selector__k': [25, 50, 100, 250],
                     'classifier__max_depth': [4, 8, 16],
                     'classifier__learning_rate': [0.001, 0.003, 0.01, 0.03],
                     'classifier__n_estimators': [128, 256, 512]}]

# These are the experiments, i.e. the selectors that are compared
selectors = [SelectKBest(get_shap_features, k=25),
             SelectKBest(mutual_info_classif, k=25),
             SelectKBest(chi2, k=25)]

for i, selector in enumerate(['SHAP', 'MutualInfo', 'Chi2']):
    select = selectors[i]
    z = select.fit_transform(X, y)
    filter = select.get_support()
    features = np.array(X.columns)
    print(f'\n\n\nselected features for {selector}:\n\n', features[filter])

# here we will store results for each selection method
clf = []
outer_val_accuracy = []
outer_val_report = []


for fs, feature_selector in enumerate(selectors):
    clf.append([])
    outer_val_accuracy.append([])
    outer_val_report.append([])
    gbc = init_model()
    mut_pipe = Pipeline([('selector', feature_selector),
                         ('classifier', gbc)])
    print(f'\n\nFeature selector nr. {fs}')
    for i in range(outer_cv):
        pct = (1 / outer_cv) * 100
        start = round(i * pct)  # define start and end index
        end = round((i + 1) * pct)

        # train and validation split
        x_val = X[start:end]
        x_train = X.drop(list(X.index[start:end]), axis=0)

        y_val = y[start:end]
        y_train = y.drop(list(y.index[start:end]), axis=0)

        # inner cross validation
        inner_cv = KFold(n_splits=8, shuffle=True, random_state=i)

        # define classifier pipeline
        clf[fs].append(GridSearchCV(estimator=mut_pipe, param_grid=mut_search_space, cv=inner_cv, n_jobs=20))
        print('\n...fitting...')
        clf[fs][i].fit(x_train, y_train.values.ravel())  # , classifier__eval_set=[(x_val, y_val.values)])
        print('...done fitting...')

        y_pred = clf[fs][i].predict(x_val)
        outer_val_accuracy[fs].append(metrics.accuracy_score(y_val, y_pred))
        outer_val_report[fs].append(classification_report(y_val, y_pred, target_names=target_names))
        print('accuracy:  ', metrics.accuracy_score(y_val, y_pred))


print("These are the best scores identified in each inner cross-validation:")
for fs, feature_selector in enumerate(['SHAP', 'Mutual Info', 'Chi2']):
    print(f'\n\n{feature_selector}:\n')
    for c, cl in enumerate(clf[fs]):
        print(f'best score of classifier {c}', cl.best_score_)
        print('And the corresponding parameters:\n', cl.best_params_)
        print('...and Classification Report :\n', outer_val_report[fs][c])

print("This is the Accuracy and Classification Report from the outer cross-validation")
for fs, feature_selector in enumerate(['SHAP', 'Mutual Info', 'Chi2']):
    print(f'\n\n{feature_selector}:\n')
    print('the Accuracy from the outer cross-validation:\n', outer_val_accuracy[fs])
    print('mean:    ', statistics.mean(outer_val_accuracy[fs]))
    print('StDev:   ', statistics.stdev(outer_val_accuracy[fs]), '\n')

print('Done!')
