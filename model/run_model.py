#!/usr/bin/env python3
"""Reproduce your result by your saved model.

This is a script that helps reproduce your prediction results using your saved
model. This script is unfinished and you need to fill in to make this script
work. If you are using R, please use the R script template instead.

The script needs to work by typing the following commandline (file names can be
different):

python3 run_model.py -i unlabelled_sample.txt -m model.pkl -o output.txt

"""

# author: Chao (Cico) Zhang
# date: 31 Mar 2017

import argparse
import sys
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn import preprocessing
import xgboost as xgb
import pickle


def main():
    '''Main function.'''
    parser = argparse.ArgumentParser(description="Reproduce the prediction")
    parser.add_argument("-i", "--input", required=True, dest="input_file",
                        metavar="unlabelled_sample.txt", type=str,
                        help="Path of the input file")
    parser.add_argument("-m", "--model", required=True, dest="model_file",
                        metavar="model.pkl", type=str,
                        help="Path of the model file")
    parser.add_argument("-o", "--output", required=True,
                        dest="output_file", metavar="output.txt", type=str,
                        help="Path of the output file")
    # Parse options
    args = parser.parse_args()

    if args.input_file is None:
        sys.exit("Input is missing!")

    if args.model_file is None:
        sys.exit("Model file is missing!")

    if args.output_file is None:
        sys.exit("Output is not designated!")

    # load sample data and preprocess (transpose, drop columns, and scale values)
    test_data = pd.read_csv(args.input_file, delimiter="\t").transpose()
    test_samples = test_data.index.values[4:]
    drop_rows = ["Chromosome", "Start", "End", "Nclone"]
    test_data = test_data.drop(index=drop_rows)

    scaler = preprocessing.MinMaxScaler()
    test_data = pd.DataFrame(scaler.fit_transform(test_data))

    models = []
    with open(args.model_file, "rb") as f:
        while True:
            try:
                models.append(pickle.load(f))
            except EOFError:
                break

    # 1st step model predictions
    y_pred = pd.DataFrame(models[0].predict(test_data))
    y_pred.index = test_data.index.values

    # remove the "0" (HER2+) labeled samples from the test set, where it is predicted by the 1st step of the model
    sub_test_data = test_data[y_pred[0] != 0]

    # use 2nd step of the model to make predictions on remaining 2 cancer subtypes
    sub_y_pred = pd.DataFrame(models[1].predict(sub_test_data))

    # +1 because new predictions will be in [0, 1] but need to be [1, 2]
    sub_y_pred = sub_y_pred + 1

    # make sure index of predictions matches that of the sub-dataset (so that original predictions can be updated)
    sub_y_pred.index = sub_test_data.index.values
    y_pred.update(sub_y_pred)
    y_pred = y_pred.astype(int)

    # now some formatting of output
    y_pred = y_pred.replace(0, "\'HER2+\'")
    y_pred = y_pred.replace(1, "\'HR+\'")
    y_pred = y_pred.replace(2, "\'Triple Neg\'")

    y_pred = y_pred.rename(columns={0: "\'Subgroup\'"})
    y_pred.index = test_samples
    y_pred = y_pred.rename(index=lambda s: "\'" + s + "\'")
    y_pred.index.name = "\'Sample\'"

    # save predictions
    y_pred.to_csv(args.output_file, sep="\t", encoding="utf-8")

    print('Done!')

if __name__ == "__main__":
    main()
