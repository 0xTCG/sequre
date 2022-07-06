""" Copied from https://github.com/brianhie/secure-dti/blob/master/bin/evaluate_demo.py"""

import datetime
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import metrics
from string import ascii_lowercase
import sys

N_HIDDEN = 1
LOSS = 'hinge'

def report_scores(X, y, W, b, act):
    y_true = []
    y_pred = []
    y_score = []
    
    for l in range(N_HIDDEN):
        if l == 0:
            act[l] = np.maximum(0, np.dot(X, W[l]) + b[l])
        else:
            act[l] = np.maximum(0, np.dot(act[l-1], W[l]) + b[l])

    if N_HIDDEN == 0:
        scores = np.dot(X, W[-1]) + b[-1]
    else:
        scores = np.dot(act[-1], W[-1]) + b[-1]

    predicted_class = np.zeros(scores.shape)
    if LOSS == 'hinge':
        predicted_class[scores > 0] = 1
        predicted_class[scores <= 0] = -1
        y = 2 * y - 1
    else:
        predicted_class[scores >= 0.5] = 1

    sys.stdout.write(str(datetime.datetime.now()) + ' | ')
    print('Batch accuracy: {}'
          .format(metrics.accuracy_score(
              y, predicted_class
          ))
    )
    
    y_true.extend(list(y))
    y_pred.extend(list(predicted_class))
    y_score.extend(list(scores))
    
    # Output aggregated scores.
    try:
        sys.stdout.write(str(datetime.datetime.now()) + ' | ')
        print('Accuracy: {0:.2f}'.format(
            metrics.accuracy_score(y_true, y_pred))
        )
        sys.stdout.write(str(datetime.datetime.now()) + ' | ')
        print('F1: {0:.2f}'.format(
            metrics.f1_score(y_true, y_pred))
        )
        sys.stdout.write(str(datetime.datetime.now()) + ' | ')
        print('Precision: {0:.2f}'.format(
            metrics.precision_score(y_true, y_pred))
        )
        sys.stdout.write(str(datetime.datetime.now()) + ' | ')
        print('Recall: {0:.2f}'.format(
            metrics.recall_score(y_true, y_pred))
        )
        sys.stdout.write(str(datetime.datetime.now()) + ' | ')
        print('ROC AUC: {0:.2f}'.format(
            metrics.roc_auc_score(y_true, y_score))
        )
        sys.stdout.write(str(datetime.datetime.now()) + ' | ')
        print('Avg. precision: {0:.2f}'.format(
            metrics.average_precision_score(y_true, y_score))
        )
    except Exception as e:
        sys.stderr.write(str(e))
        sys.stderr.write('\n')
        
    return y_true, y_pred, y_score


def load_model():
    W = [ [] for _ in range(N_HIDDEN + 1) ]
    for l in range(N_HIDDEN+1):
        W[l] = np.loadtxt(f'results/drug_target_interaction_inference_results_weights_layer_{l}.txt')

     # Initialize bias vector with zeros.
    b = [ []  for _ in range(N_HIDDEN + 1) ]
    for l in range(N_HIDDEN+1):
        b[l] = np.loadtxt(f'results/drug_target_interaction_inference_results_bias_layer_{l}.txt')

    # Initialize activations.
    act = [ [] for _ in range(N_HIDDEN) ]

    return W, b, act
    
if __name__ == '__main__':
    W, b, act = load_model()

    X_train = np.loadtxt('tests/data/dti/input/features.txt')
    y_train = np.loadtxt('tests/data/dti/input/labels.txt')
    
    print('Training accuracy:')
    report_scores(X_train, y_train, W, b, act)

    X_test = np.loadtxt('tests/data/dti/input/test_features.txt')
    y_test = np.loadtxt('tests/data/dti/input/test_labels.txt')
    
    print('Testing accuracy:')
    report_scores(X_test, y_test, W, b, act)
