"""
This script predict the breast cancer proliferation score for the input slide images
"""
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import re
import math
import numpy as np
import scipy
from scipy.stats import randint, uniform
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, make_scorer
from pyspark.sql import SparkSession


SLIDE_ID = 'slide_id'
ROI_ID = 'ROI_ID'
MITOSIS_NUM_PER_ROI = 'mitosis_num_per_ROI'
ROW_NUM = 'row_num'
COL_NUM = 'col_num'
SCORE = 'score'
MITOSIS_SCORE = 'mitosis_score'
MOLECULAR_SCORE = 'molecular_score'

FEATURE_COLUMNS = ['sum', 'std', 'avg',
                   'sum_top10', 'std_top10', 'avg_top10',
                   'sum_top30_70', 'std_top30_70', 'avg_top30_70']

def load_data_X_in_batch(file_paths):
  """ Generate the X dataframe from a list of csv files

  Args:
    file_paths: list of csv file paths for X

  Return:
    a dataframe for X
  """
  frames = [load_data_X(file_path) for file_path in file_paths]
  return pd.concat(frames)

def load_data_X(file_path):
  X = pd.read_csv(file_path, index_col=0, skiprows=[0],
                  names=[SLIDE_ID, ROI_ID, MITOSIS_NUM_PER_ROI, ROW_NUM, COL_NUM, SCORE])
  X[MITOSIS_NUM_PER_ROI] = X[MITOSIS_NUM_PER_ROI].apply(pd.to_numeric, errors='coerce')
  X[ROW_NUM] = X[ROW_NUM].apply(pd.to_numeric, errors='coerce')
  X[COL_NUM] = X[COL_NUM].apply(pd.to_numeric, errors='coerce')
  X[SCORE] = X[SCORE].apply(pd.to_numeric, errors='coerce')
  X[SLIDE_ID] = X[SLIDE_ID].apply(lambda x: int(re.findall("\d+", x)[0]))

  return X

def load_data_Y(file_path):
  """ Generate the Y dataframe from the input csv file

  Args:
    file_path: csv file path for Y

  Return:
    a dataframe for Y
  """
  Y = pd.read_csv(file_path, names=[MITOSIS_SCORE, MOLECULAR_SCORE])
  Y[MITOSIS_SCORE] = Y[MITOSIS_SCORE].apply(pd.to_numeric, errors='coerce')
  Y[MOLECULAR_SCORE] = Y[MOLECULAR_SCORE].apply(pd.to_numeric, errors='coerce')
  Y.insert(0, SLIDE_ID, range(1, len(Y)+1))
  return Y


def generate_model_input_data(x_files, y_file, test_size, selected_feature_names, y_name,
                              seed=None):
  """ Prepare the input data for the training model

  Args:
    x_files: input file paths for X.
    y_file: input file paths for y.
    test_size: ratio of input data for the test.
    selected_feature_names: training features selected for the SVM model.
    y_name: feature name selected for target values.
    seed: integer seed for the train and test data split shuffling.

  Return:
    the dataframe for the training and test.
  """
  X = load_data_X_in_batch(x_files)
  Y = load_data_Y(y_file)

  features, feature_col_names = gen_feature_table(X)
  datasets = join_df([features, Y], SLIDE_ID)
  datasets.to_csv("svs_prediction.csv")

  assert set(selected_feature_names).issubset(feature_col_names)
  X_clean = datasets[selected_feature_names].values
  Y_clean = datasets[y_name].values

  X_train, X_test, y_train, y_test = train_test_split(X_clean, Y_clean, shuffle=True,
                                                      test_size=test_size,
                                                      random_state=seed)

  return X_train, X_test, y_train, y_test

def visualize(datasets, columns):
  """ Visualize the selected columns in the

  Args:
    datasets: dataframe for the visualization
    columns: column names for the visualization
  """
  datasets[columns].plot()
  plt.show()


def gen_feature_table(df):
  """ Generate the features from the input prediction data

  Args:
    df: dataframe of the input prediction result

  Return:
    the dataframe of the features for the model training
  """
  feature_col_names = []

  # the schema for the input df is as [slide_id, roi_id, mitosis_num_per_roi,
  # row_num, col_num, score]; Each row in the df represents a detected
  # mitosis point. Grouping by the slide_id and roi_id, we will get the
  # number of mitoses for each ROI.
  mitosis_df = df.groupby([SLIDE_ID, ROI_ID])[MITOSIS_NUM_PER_ROI].count().reset_index(
    name=MITOSIS_NUM_PER_ROI)

  # generate the features (sum, std, avg) all the ROIs
  mitosis_sum, mitosis_std, \
  mitosis_avg, feature_names = get_descriptive_statistics(mitosis_df,
                                                          SLIDE_ID,
                                                          MITOSIS_NUM_PER_ROI, suffix="")
  # collect the name of features
  feature_col_names += feature_names

  # get the top 10% ROIs with the largest number of mitoses for each slide
  mitosis_df_top10 = mitosis_df.groupby([SLIDE_ID]).apply(
    lambda row: row.sort_values([MITOSIS_NUM_PER_ROI], ascending=False)[
                0: max(int(len(row) * 0.1), 1)])

  # generate the features (sum, std, avg) for the top 10% ROIs
  mitosis_sum_top10, mitosis_std_top10, \
  mitosis_avg_top10, feature_names = get_descriptive_statistics(mitosis_df_top10, SLIDE_ID,
                                                                MITOSIS_NUM_PER_ROI,
                                                                suffix="top10")
  # collect the name of features
  feature_col_names += feature_names

  # get the top 30%  - 70% ROIs with the largest number of mitoses for
  # each slide
  mitosis_df_top30_70 = mitosis_df.groupby([SLIDE_ID]).apply(
    lambda row: row.sort_values([MITOSIS_NUM_PER_ROI], ascending=False)[
                int(len(row) * 0.3): max(int(len(row) * 0.7), 1)])

  # generate the features (sum, std, avg) for the top 30-70% ROIs
  mitosis_sum_top30_70, mitosis_std_top30_70, \
  mitosis_avg_top30_70, feature_names = get_descriptive_statistics(mitosis_df_top30_70, SLIDE_ID,
                                                                   MITOSIS_NUM_PER_ROI,
                                                                   suffix="top30_70")
  # collect the name of features
  feature_col_names += feature_names

  # join these tree group of features together by slide_id
  feature_df = join_df([mitosis_sum, mitosis_std, mitosis_avg,
                        mitosis_sum_top10, mitosis_std_top10, mitosis_avg_top10,
                        mitosis_sum_top30_70, mitosis_std_top30_70, mitosis_avg_top30_70],
                       SLIDE_ID)

  return feature_df, feature_col_names


def get_descriptive_statistics(df, groupby_col, target_col, suffix=""):
  """Compute the descriptive statistics (sum, std, mean) for the input
    features

  Args:
    df: dataframe for the input prediction data
    groupby_col: column name used to group the table
    target_col: column name for the descriptive statistics
    suffix: suffix for the new statistic features

  Return:
    the dataframes for the descriptive statistics
  """
  sum_col_name = f"sum_{suffix}" if len(suffix) > 0 else "sum"
  std_col_name = f"std_{suffix}" if len(suffix) > 0 else "std"
  avg_col_name = f"avg_{suffix}" if len(suffix) > 0 else "avg"

  sum = df.groupby([groupby_col])[target_col].sum().reset_index(name=sum_col_name)
  std = df.groupby([groupby_col])[target_col].std().reset_index(name=std_col_name)
  avg = df.groupby([groupby_col])[target_col].mean().reset_index(name=avg_col_name)

  # handle the special case that the std df only has one row, pd will
  # set the value to be NaN. Here we set it to be 0, which will be
  # compatible with the SVM algorithm.
  std[std_col_name] = std[std_col_name].apply(lambda x : 0 if math.isnan(x) else x)

  return sum, std, avg, [sum_col_name, std_col_name, avg_col_name]


def join_df(df_list, key):
  """ Join the dataframes by the key column

  Args:
    df_list: list of dataframes for joining
    key: column name to join on

  Return:
    the merged dataframe
  """
  left = df_list[0]
  for right in df_list[1:]:
    left = pd.merge(left, right, on=key)
  return left

def compute_kappa_score(y_truth, y_pred, weights="quadratic"):
  """Compute the Cohen's kappa score between the prediction and the
    ground truth

  Args:
    y_pred: prediction results.
    y_truth: ground truth.
    weights: string of weighting type to calculate the score. None
      means no weighted; “linear” means linear weighted; “quadratic”
      means quadratic weighted.

  Return:
    the kappa score, which is a float number between -1 and 1.
  """
  kappa_score = cohen_kappa_score(y_pred, y_truth, weights=weights)
  return kappa_score


def report(results, n_top=3):
  """report the model training result

  Args:
    results: cross validation result with keys as column headers and
      values as columns
    n_top: number of the top-ranked models
  """
  for i in range(1, n_top + 1):
    candidates = np.flatnonzero(results['rank_test_score'] == i)
    for candidate in candidates:
      print("Model with rank: {0}".format(i))
      print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
        results['mean_test_score'][candidate],
        results['std_test_score'][candidate]))
      print("Parameters: {0}".format(results['params'][candidate]))
      print("")


def get_scoring(score):
  """ get the scoring function to evaluate the prediction on the test set
  Args:
    score: string for the scoring method

  Return:
    the scoring function
  """
  if score == 'kappa':
    return make_scorer(compute_kappa_score, greater_is_better=True)
  else:
    raise Exception(f"scoring mode unknown: {score}")


def search_hyper_parameter_help(parameters, parameter_search_mode, x_files, y_file,
                                selected_feature_names,
                                y_name, test_size,
                                scores, cv=10, seed=None, n_iter=2000):
  """Help function for searching the hyper parameters of the SVM model

  Args:
    parameters: hyper parameters for SVM model, e.g. [{'kernel': ['rbf'],
     'gamma': [0.8, 0.7, 0.6], 'C': [1, 2, 3]}, {'kernel': ['linear'],
     'C': [1, 2, 3]}].
    parameter_search_mode: mode for hyper parameter searching, e.g. random/grid
    x_files: input file paths for X.
    y_file: input file path for y.
    selected_feature_names: training features selected for the SVM model.
    y_name: feature name selected for target values.
    test_size: ratio of input data for the test.
    scores: list of strings for the evaluation of predictions on the test
      set, e.g., ['precision', 'recall'].
    cv: integer to determine the cross-validation splitting strategy.
    seed: integer seed for the train and test data split shuffling.
    n_iter: number of parameter settings that are sampled.

  Return:
    the model evaluation result, a tuple (clf.best_score_, test_kappa_score, clf.best_params_).
  """
  # generate the training and test datasets from the input raw dataset
  X_train, X_test, y_train, y_test = generate_model_input_data(x_files=x_files, y_file=y_file,
                                                               test_size=test_size,
                                                               selected_feature_names=selected_feature_names,
                                                               y_name=y_name, seed=seed)
  scoring = get_scoring(scores)

  if parameter_search_mode == "grid":
    clf = GridSearchCV(SVC(), param_grid=parameters, scoring=scoring, cv=cv)
  elif parameter_search_mode == "random":
    clf = RandomizedSearchCV(SVC(), param_distributions=parameters[0], n_iter=n_iter, cv=cv,
                             scoring=scoring)
  else:
    raise Exception(f"hyper parameter searching mode unknown: {parameter_search_mode}")

  clf.fit(X_train, y_train)

  # print the information for the n_top models
  report(clf.cv_results_, n_top=3)

  # compute the kappa score of the best model on the validation dataset
  y_true, y_pred = y_test, clf.predict(X_test)
  test_kappa_score = compute_kappa_score(y_true, y_pred)

  result = (clf.best_score_, test_kappa_score, clf.best_params_)

  return result


def search_hyper_parameter(sparkContext, parameters, parameter_search_mode, score, x_files, y_file,
                           selected_feature_names, y_name, test_size = 0.2, cv=10,
                           seed=None, n_iter=2000):
  """ Utilize PySpark to search the hyper parameters of the SVM model

  Args:
    sparkContext: Spark context.
    parameters: hyper parameters for SVM model, e.g. [{'kernel': ['rbf'],
      'gamma': [0.8, 0.7, 0.6], 'C': [1, 2, 3]}, {'kernel': ['linear'],
      'C': [1, 2, 3]}].
    parameter_search_mode: mode for hyper parameter searching, e.g. random/grid
    score: list of strings for the evaluation of predictions on the
      test set, e.g., ['precision', 'recall'].
    x_files: input file paths for X.
    y_file: input file path for y.
    selected_feature_names: selected training features for the SVM model.
    y_name: selected feature name for target values.
    test_size: ratio of input data for the test.
    cv: integer to determine the cross-validation splitting strategy.
    seed: integer seed for the train and test data split shuffling.
    n_iter: number of parameter settings that are sampled.

  Return:
    the model evaluation results.
  """
  model_intputs = [([parameter], x_files, y_file) for parameter in parameters]
  para_rdd = sparkContext.parallelize(model_intputs, len(model_intputs))
  model_evaluation_rdd = para_rdd.map(lambda input: search_hyper_parameter_help(
                                                parameters=input[0],
                                                parameter_search_mode=parameter_search_mode,
                                                x_files=input[1], y_file=input[2],
                                                selected_feature_names=selected_feature_names,
                                                y_name=y_name, test_size=test_size, scores=score,
                                                cv=cv, seed=seed, n_iter=n_iter))
  model_evaluation_results = model_evaluation_rdd.collect()
  return model_evaluation_results


def main(args=None):
  # parse args
  parser = argparse.ArgumentParser()
  parser.add_argument("--appName", default="Proliferation Score Prediction",
                      help="application name")
  parser.add_argument("--ground_truth_file_path", required=True,
                      help="path to the ground truth file")
  parser.add_argument("--pred_file_paths", required=True, nargs='*',
                      help="list of paths for the prediction results ")
  parser.add_argument("--cv", type=int, default=10,
                      help="integer to determine the cross-validation splitting strategy")
  parser.add_argument("--score", default='kappa',
                      help="string for the evaluation of predictions on the test set ("
                           "default: %(default)s).")
  parser.add_argument("--test_data_size", type=float, default=0.2,
                      help="float ratio of the test data")
  parser.add_argument("--seed", type=int, default=0,
                      help="integer seed for the train and test data split shuffling")
  parser.add_argument("--y_name", default=MITOSIS_SCORE,
                      help="column name for Y, (default: %(default)s)")
  parser.add_argument("--parameter_search_mode", default="random",
                      help="hyper parameter searching mode (e.g. random/grid), (default: %("
                           "default)s)")
  parser.add_argument("--n_iter", type=int, default=2000,
                      help="number of parameter settings that are sampled for RandomizedSearchCV")

  args = parser.parse_args(args)

  spark = (SparkSession.builder
           .appName(args.appName)
           .getOrCreate())

  sparkContext = spark.sparkContext

  selected_feature_names = FEATURE_COLUMNS[0:len(FEATURE_COLUMNS)]

  # Set the parameters by cross-validation
  if args.parameter_search_mode == "grid":
    tuned_parameters = [{'kernel': ['rbf'],
                         'gamma': [1, 2, 3, 0.1, 0.2, 0.3],
                         'C': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                         'class_weight': ['balanced', None]},
                        {'kernel': ['rbf'],
                         'gamma': [1, 2, 3, 0.1, 0.2, 0.3],
                         'C': [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
                         'class_weight': ['balanced', None]},
                        {'kernel': ['linear'],
                         'C': [1]}]
  elif args.parameter_search_mode == "random":
    tuned_parameters = [{'C': scipy.stats.expon(loc=1, scale=10),
                        'gamma': scipy.stats.expon(loc=1, scale=10),
                        'kernel': ['rbf'],
                        'class_weight': ['balanced', None]},
                        {'C': scipy.stats.expon(loc=10, scale=10),
                         'gamma': scipy.stats.expon(loc=0.1, scale=10),
                         'kernel': ['rbf'],
                         'class_weight': ['balanced', None]},
                        {'C': scipy.stats.expon(loc=0.1, scale=10),
                         'gamma': scipy.stats.expon(loc=0.01, scale=10),
                         'kernel': ['rbf'],
                         'class_weight': ['balanced', None]},
                        {'C': uniform(loc=0.001, scale=0.009),
                         'gamma': uniform(loc=0.1, scale=0.9),
                         'kernel': ['rbf'],
                         'class_weight': ['balanced', None]},
                        {'C': uniform(loc=0.001, scale=0.009),
                         'gamma': uniform(loc=0.01, scale=0.09),
                         'kernel': ['rbf'],
                         'class_weight': ['balanced', None]}]
  else:
    raise Exception(f"hyper parameter searching mode unknown: {args.parameter_search_mode}")

  model_evaluation_results = search_hyper_parameter(sparkContext=sparkContext,
                                                    parameters=tuned_parameters,
                                                    parameter_search_mode=args.parameter_search_mode,
                                                    score=args.score,
                                                    x_files=args.pred_file_paths,
                                                    y_file=args.ground_truth_file_path,
                                                    selected_feature_names=selected_feature_names,
                                                    y_name=args.y_name,
                                                    test_size=args.test_data_size,
                                                    cv=args.cv, seed=args.seed,
                                                    n_iter=args.n_iter)
  print(model_evaluation_results)

if __name__ == "__main__":
  main()
