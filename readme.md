# Group 16 code submissions

This readme is to briefly explain what each script is meant to do. For all scripts to run as is, the data (provided in the "data" folder with this submission) must be stored in a directory named "data". All dependencies for the python code can be found in requirements.txt.

We include the following .py scripts + 1 .R script:
- feature_selection.py:

      This script selects the k-best features using our sleected methods and outputs the features without ordering.

- cv1.py:

      This Script runs our first cross validation experiment, where we test just our gradient boosting classifier (initial single step model).

- 2step.py

      This script runs our second cross validation experiment, where we test our 2-step model. Some Plotting is also performed.

- baseline.py

      This script runs our baselines and creates a plot comparing ot our best performing model.

- plot.py

      This script creates a unfied plot for our cross validation results.

- final_model.py

      This script creates and saves our model. It is built with the best hyperparameters and feature selection method. 

- som.R

      This script is used to build the four different Self-Organazing Maps (SOM). It requires the R packages: kohonen, tidyr, dplyr and readr
