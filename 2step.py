## Imports
import random
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import statistics
from collections import Counter


np.random.seed(42)
random.seed(42)

# load data
def load_data():
    x_vals = pd.read_csv('data/train_call.txt', delimiter='\t')
    y_vals = pd.read_csv('data/train_clinical.txt', delimiter='\t')
    return x_vals.transpose(), y_vals


# Define base model
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

# Define 2nd-step model, only relevant for feature selection, because we use it for getting SHAP features
def init_model2():
    classifier2 = xgb.XGBClassifier(
      booster='gbtree',
      learning_rate=.01,
      max_delta_step=0,
      max_depth=16,
      min_child_weight=1,
      n_estimators=256,
      objective='binary:logistic',
      random_state=0,
      eval_metric=accuracy_score,
    )
    return classifier2


""" SHAP-BASED FEATURE SELECTION"""
# it would be simple enough to extract most relevant features, but because of a
# naive approach where we select the features after the initial modeling, it
# makes a difference which train-validation-split we take. Therefore we do an
# initial cross-validation step to identify which features most commonly appear.
# This is done by simply taking the most frequently+highly ranked features

def get_shap_features2(X, y, shap_cv=25):
    # we will add reversed rank of features to this list, to also take into account feature importance
    scores = [0] * X.shape[1]

    # start CV loop
    for i in range(shap_cv):
        gbc_shap = init_model2()
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


""" NESTED CROSS VALIDATION """
# Define nested cross-validation parameters (NCV)
outer_cv = 50

# Define model parameter search space
search_space = [{'selector__k': [50, 100, 250, 500, 1000],
                 'classifier__learning_rate_init': [0.0003, 0.001, 0.003, 0.01]}]

# These are the experiments, i.e. the selectors that are compared
selectors = [SelectKBest(get_shap_features2, k=25),
             SelectKBest(mutual_info_classif, k=25),
             SelectKBest(chi2, k=25)]

gbc = init_model()
# here we just choose the fastest feature selection (because all methods work well with GBC anyway) and get the predictions for HER2+
pipe1 = Pipeline([('selector', SelectKBest(chi2, k=500)),
                 ('classifier', gbc)])

# here we will store results
clf_1 = []
clf_2 = []
outer_val_accuracy1 = []    # outer validation accuracies after 1st step
outer_val_accuracy2 = []    # outer validation accuracies after 2nd step
outer_val_report = []
all_y, all_pred = [], []

for fs, feature_selector in enumerate(selectors):

    print(f'\n\nFeature selector nr. {fs}')
    clf_1.append([])
    clf_2.append([])
    outer_val_accuracy1.append([])
    outer_val_accuracy2.append([])
    outer_val_report.append([])
    all_y.append([])
    all_pred.append([])

    pipe2 = Pipeline([('selector', feature_selector),
                      ('classifier', MLPClassifier(alpha=0.05, max_iter=2500))])  # GaussianProcessClassifier(5.0 * RBF(10.0)))])
    test_sizes = [0.14, 0.17, 0.2, 0.23]

    for i in range(outer_cv):
        """ Initial Data Split (for outer cross validation) """
        # train and validation split
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=random.choice(test_sizes), random_state=i)

        # inner cross validation
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=i)

        # define classifier pipeline
        clf_1[fs].append(pipe1)  # GridSearchCV(estimator=pipe, param_grid=search_space, cv=inner_cv, n_jobs=20))
        print('\n...fitting...')
        clf_1[fs][i].fit(x_train, y_train.values.ravel())  # , classifier__eval_set=[(x_val, y_val.values)])
        print('...done fitting...')

        # check if this initial run indeed predicts all HER2+ labels correctly
        y_pred = pd.DataFrame(clf_1[fs][i].predict(x_val))
        y_pred.index = y_val.index.values
        print('\nClassification Report:\n', classification_report(y_val, y_pred, target_names=target_names))
        outer_val_accuracy1[fs].append(metrics.accuracy_score(y_val, y_pred))

        # Now we remove the "0" labeled samples from the train set
        x_sub_train = x_train[y_train[0] != 0]
        y_sub_train = y_train[y_train[0] != 0]

        # ...and the "0" labeled samples from the test set, where it is predicted by the initial model
        x_sub_val = x_val[y_pred[0] != 0]
        y_sub_val = y_val[y_pred[0] != 0]

        # re-labeling (because we removed all zeros)
        y_sub_train = y_sub_train - 1

        # and now we train the second model on the remaining data
        clf_2[fs].append(GridSearchCV(estimator=pipe2, param_grid=search_space, cv=inner_cv, n_jobs=20))
        print('\n...Gridsearch & fitting...')
        clf_2[fs][i].fit(x_sub_train, y_sub_train.values.ravel())  # , classifier__eval_set=[(x_val, y_val.values)])
        print('...done fitting...')

        # now check if this run predicts the other labels better
        y_sub_pred = pd.DataFrame(clf_2[fs][i].predict(x_sub_val))
        y_sub_pred = y_sub_pred + 1  # +1 to obtain original classes
        y_sub_pred.index = y_sub_val.index.values
        y_pred.update(y_sub_pred); y_pred = y_pred.astype(int)
        print('\n2nd-Step Classification Report:\n', classification_report(y_val, y_pred, target_names=target_names))
        print('And the corresponding parameters:\n', clf_2[fs][i].best_params_)
        outer_val_accuracy2[fs].append(metrics.accuracy_score(y_val, y_pred))
        outer_val_report[fs].append(classification_report(y_val, y_pred, target_names=target_names))
        all_y[fs].extend(y_val[0].to_list())
        all_pred[fs].extend(y_pred[0].to_list())


for fs, feature_selector in enumerate(['SHAP', 'Mutual Info', 'Chi2']):
    print(f'\n\n{feature_selector}:\n')
    for c, cl in enumerate(clf_2[fs]):
        print(f'best score of classifier {c}', cl.best_score_)
        print('And the corresponding parameters:\n', cl.best_params_)
    params = [tuple(cl.best_params_.values()) for cl in clf_2[fs]]
    params_counter = Counter(params)
    best_params = params_counter.most_common(1)
    print('\nBest Paramaters:\n', best_params)

    print("These are the Accuracies from the outer cross-validation")
    print('1st Step Accuracy from the outer cross-validation:\n', outer_val_accuracy1[fs])
    print('Mean:    ', statistics.mean(outer_val_accuracy1[fs]))
    print('StDev:   ', statistics.stdev(outer_val_accuracy1[fs]), '\n')
    print('2nd Step Accuracy from the outer cross-validation:\n', outer_val_accuracy2[fs])
    print('Mean:    ', statistics.mean(outer_val_accuracy2[fs]))
    print('StDev:   ', statistics.stdev(outer_val_accuracy2[fs]), '\n')

    # Confusion matrix of results for all outer validation runs
    cm = confusion_matrix(np.array(all_y[fs]), np.array(all_pred[fs]))
    ax = sns.heatmap(cm, annot=True, cmap='YlGnBu', fmt='.4g')

    ax.set_title(f'Confusion Matrix of Predicted vs True labels\nwith {feature_selector} Feature Selection')
    ax.set_xlabel('\nPredicted Labels')
    ax.set_ylabel('True Labels ')

    # Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['HER2+', 'HR+', 'Triple Neg'])
    ax.yaxis.set_ticklabels(['HER2+', 'HR+', 'Triple Neg'])
    plt.show()

    # This is for getting individual plots for each feature selection method.
    # There is a separate script for getting a combined plot
    # input data:
    befores = np.array(outer_val_accuracy1[fs])
    afters = np.array(outer_val_accuracy2[fs])

    plt.boxplot([befores, afters])

    # plotting the points
    plt.scatter(np.ones(len(befores)), befores)
    plt.scatter(np.ones(len(afters))+1, afters)

    # uncomment next two line for plotting lines between paired runs (i.e. how does accuracy improve for each run)
    # for i in range(len(befores)):
    #     plt.plot([1, 2], [befores[i], afters[i]], c='k')

    plt.xticks([1, 2], ['1st step accuracy', '2nd step accuracy'])
    plt.title(f'{feature_selector} Feature Selection\n1st step vs 2nd step Accuracies')
    plt.show()

print('\n\nDone!')
