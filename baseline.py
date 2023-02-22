## Imports
import random
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import statistics


np.random.seed(42)
random.seed(42)

# load data
def load_data():
    x_vals = pd.read_csv('data/train_call.txt', delimiter='\t')
    y_vals = pd.read_csv('data/train_clinical.txt', delimiter='\t')
    return x_vals.transpose(), y_vals


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

# here we just choose the fastest feature selection (because all methods work well with GBC anyway) and get the predictions for HER2+
svm1 = Pipeline([('selector', SelectKBest(chi2, k=250)),
                 ('classifier', SVC(kernel="linear"))])
svm2 = Pipeline([('selector', SelectKBest(chi2, k=250)),
                 ('classifier', SVC(kernel="rbf"))])

pipes = [None, svm1, svm2]

# here we will store results
clf = []
outer_val_accuracy1 = []  # outer validation accuracies after 1st step
outer_val_accuracy2 = []  # outer validation accuracies after 2nd step
outer_val_report = []
all_y, all_pred = [], []

for fs, base in enumerate(['Random', 'SVM', 'SVM - Radial Kernel']):

    print(f'\n\nBaseline - {base}')

    clf.append([])
    outer_val_accuracy1.append([])
    outer_val_accuracy2.append([])
    outer_val_report.append([])
    all_y.append([])
    all_pred.append([])

    test_sizes = [0.14, 0.17, 0.2, 0.23]

    for i in range(outer_cv):
        """ Initial Data Split (for outer cross validation) """
        # train and validation split
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=random.choice(test_sizes), random_state=i)

        if base != 'Random':
            # define classifier pipeline
            clf[fs].append(pipes[fs])
            print('\n...fitting...')
            clf[fs][i].fit(x_train, y_train.values.ravel())
            print('...done fitting...')

            # check if this initial run indeed predicts all HER2+ labels correctly
            y_pred = pd.DataFrame(clf[fs][i].predict(x_val))
        else:
            y_pred = pd.DataFrame([random.choice([0, 1, 2]) for x in range(y_val.shape[0])])

        print('\nClassification Report:\n', classification_report(y_val, y_pred, target_names=target_names))
        outer_val_accuracy1[fs].append(metrics.accuracy_score(y_val, y_pred))


for fs, baseline in enumerate(['Random', 'SVM', 'SVM - Radial Kernel']):
    print(f'\n\n{baseline}:\n')

    print("These are the Accuracies from the outer cross-validation")
    print('1st Step Accuracy from the outer cross-validation:\n', outer_val_accuracy1[fs])
    print('Mean:    ', statistics.mean(outer_val_accuracy1[fs]))
    print('StDev:   ', statistics.stdev(outer_val_accuracy1[fs]), '\n')

# we add our best-performing model + feature selection accuracies here for comparison
chi2_2 = [0.8695652173913043, 0.8695652173913043, 0.9, 0.8666666666666667, 0.8, 0.8, 0.95, 0.8, 0.85, 0.9130434782608695, 0.8235294117647058, 0.8695652173913043, 0.8666666666666667, 0.85, 0.7058823529411765, 1.0, 1.0, 0.8823529411764706, 0.8235294117647058, 0.85, 0.8823529411764706, 0.9333333333333333, 0.85, 0.7391304347826086, 0.9333333333333333, 0.9333333333333333, 0.8, 0.8, 0.7058823529411765, 0.8666666666666667, 0.8823529411764706, 0.9333333333333333, 0.8666666666666667, 0.8260869565217391, 0.8666666666666667, 0.8235294117647058, 0.8823529411764706, 0.8695652173913043, 0.8235294117647058, 1.0, 0.782608695652174, 0.8235294117647058, 0.8235294117647058, 0.85, 0.9130434782608695, 0.85, 0.8695652173913043, 0.8695652173913043, 0.9333333333333333, 0.8235294117647058]
df = pd.DataFrame(list(zip(outer_val_accuracy1[0], outer_val_accuracy1[1], outer_val_accuracy1[2], chi2_2)), columns=['Random', 'SVM - Linear Kernel', 'SVM - Radial Kernel', 'Chi-squared\n2-step Model'])

ax = sns.boxplot(data=df, palette='Set3', showfliers=False)
sns.stripplot(
    data=df,
    dodge=True,
    jitter=True,
    alpha=0.5
).set(title='Comparison of Baseline Accuracies')
plt.show()

print('\n\nDone!')
