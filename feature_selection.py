import pandas as pd
import numpy as np

#univariate

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression, mutual_info_regression, f_classif, mutual_info_classif


#For regression: f_regression, mutual_info_regression
#For classification: chi2, f_classif, mutual_info_classif
def feature_fitness_selection(X,Y, metric):
    test = SelectKBest(score_func= metric, k=4)
    fit = test.fit(X, Y)
    numpy.set_printoptions(precision=3)
    print(fit.scores_)
    features = fit.transform(X)
    return(features)


#RFE
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_mode import LassoCV, LogisticRegression

#For regression: lasso
#For classification: Logit, SVC

def ref_feature_selection(X, Y, target_class):
    if "reg" in target_class:
        model = LassoCV()
        sfm = SelectFromModel(model, threshold=0.25)
        sfm.fit(X, y)
        n_features = sfm.transform(X).shape[1]

        # Reset the threshold till the number of features equals two.
        while n_features > 2:
            sfm.threshold += 0.1
            X_transform = sfm.transform(X)
            n_features = X_transform.shape[1]
            print("Number of features: %s" n_features)
            return(X_transform)
    elif "class" in target_class:
        
        
