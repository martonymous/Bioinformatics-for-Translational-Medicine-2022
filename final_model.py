## Imports
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn import preprocessing
import xgboost as xgb
import shap
from pickle import dump
import numpy as np


# Set true if shap visualization is to be made (in which case model is not saved)
SHAP_VIS = False

# load data
def load_data():
    x_vals = pd.read_csv('data/train_call.txt', delimiter='\t')
    y_vals = pd.read_csv('data/train_clinical.txt', delimiter='\t')
    x_test = pd.read_csv('data/Validation_call.txt', delimiter='\t')
    return x_vals.transpose(), y_vals, x_test.transpose()


# Define 1st step of the model
def init_model():
    classifier = xgb.XGBClassifier(
      booster='gbtree',
      learning_rate=.01,
      max_delta_step=0,
      max_depth=16,
      min_child_weight=1,
      n_estimators=256,
      objective='multi:softmax',
      random_state=0,
      eval_metric=accuracy_score,
    )
    return classifier


""" PRE-PROCESSING and MODEL DEFINITIONS """
# get data and drop unusable/irrelevant columns
X, y, x_test = load_data()
test_samples = x_test.index.values[4:]
drop_rows = ['Chromosome', 'Start', 'End', 'Nclone']
X = X.drop(index=drop_rows)
x_test = x_test.drop(index=drop_rows)
y = y.set_index('Sample')

# encode labels (HER2+ = 0, HR+ = 1, Triple Neg = 2)
target_names = ['HER2+', 'HR+', 'Triple Neg']
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(y)
y = pd.DataFrame(label_encoder.transform(y))

# Define preprocessing steps, just scaling in this case
scaler = preprocessing.MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X))
x_test = pd.DataFrame(scaler.fit_transform(x_test))

# 1st step
gbc = init_model()
pipe1 = Pipeline([('selector', SelectKBest(chi2, k=500)),
                 ('classifier', gbc)])

# 2nd step, includes definition of the neural network (MLP) used
if SHAP_VIS:
    pipe2 = MLPClassifier(alpha=0.05, max_iter=2500, learning_rate_init=0.0003)
else:
    pipe2 = Pipeline([('selector', SelectKBest(chi2, k=250)),
                      ('classifier', MLPClassifier(alpha=0.05, max_iter=2500, learning_rate_init=0.0003))])


""" MODELING """
# fit and explain 1st step model
pipe1.fit(X, y.values.ravel())
y_pred = pd.DataFrame(pipe1.predict(x_test))
y_pred.index = x_test.index.values

if SHAP_VIS:
    select = SelectKBest(chi2, k=250)
    z = select.fit_transform(X, y)
    filters = select.get_support()
    features = np.array(X.columns)

    # use SHAP explainer to explain 1st step predictions
    explainer = shap.TreeExplainer(gbc)
    shap_values = explainer.shap_values(x_test[features[filters]])

    shap.summary_plot(shap_values, X[features[filters]])

# Now we remove the "0" labeled samples from the train set
x_sub_train = X[y[0] != 0]
y_sub_train = y[y[0] != 0]

# and relabel
y_sub_train = y_sub_train - 1

# ...and the "0" labeled samples from the test set, where it is predicted by the initial model
x_sub_test = x_test[y_pred[0] != 0]

if SHAP_VIS:
    # 2nd step feature selection (done outside of pipeline function to enable use with SHAP)
    select = SelectKBest(chi2, k=250)
    z = select.fit_transform(x_sub_train, y_sub_train)
    filters = select.get_support()
    features = np.array(x_sub_train.columns)

    # fit  2nd step model
    pipe2.fit(x_sub_train[features[filters]], y_sub_train.values.ravel())
    y_sub_pred = pd.DataFrame(pipe2.predict(x_sub_test[features[filters]]))

    # explain model predictions
    explainer = shap.KernelExplainer(pipe2.predict_proba, x_sub_train[features[filters]])
    shap_values = explainer.shap_values(x_sub_test[features[filters]])

    shap.summary_plot(shap_values, x_sub_train[features[filters]])

else:
    # fit  2nd step model
    pipe2.fit(x_sub_train, y_sub_train.values.ravel())
    y_sub_pred = pd.DataFrame(pipe2.predict(x_sub_test))

y_sub_pred = y_sub_pred + 1  # +1 to obtain original classes
y_sub_pred.index = x_sub_test.index.values
y_pred.update(y_sub_pred)
y_pred = y_pred.astype(int)

# now some formatting of output
y_pred = y_pred.replace(0, '\'HER2+\'')
y_pred = y_pred.replace(1, '\'HR+\'')
y_pred = y_pred.replace(2, '\'Triple Neg\'')

y_pred = y_pred.rename(columns={0: "\'Subgroup\'"})

""" SAVE PREDICTIONS AND MODELS """
y_pred.index = test_samples
y_pred = y_pred.rename(index=lambda s: "\'" + s + "\'")
y_pred.index.name = "\'Sample\'"
y_pred.to_csv("output.txt", sep='\t')

with open("model.pkl", "wb") as f:
    dump(pipe1, f)
    dump(pipe2, f)

print('Done!')
