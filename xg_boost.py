#Read in packages needed to do the modeling
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor #note: Xcode doesnt support OpenMP multi-threading, which XGBoost uses. To fix this you run 'brew install gcc@5' in the terminal to fix.
import xgboost as xgb
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectKBest
#create the preprocessing transformer feature
from preprocessing import preprocessing_transformer

#feature selection pipeline 
feature_selection_pipeline = Pipeline(FeatureUnion([
    ('select_k_best', SelectKBest(k=10))#more FE transformers to come later
    ]))



#Classifier
xg_classifier_pipe = Pipeline([
    ('preprocessing', preprocessing_transformer),
    ('feature_selection', feature_selection_pipeline),
    ('xgb', xgb.XGBClassifier(learning_rate=0.1, #placeholder values, need to search over the param grid 
                              n_estimators=100, 
                              objective="binary:logistic",
                              eval_met
                              seed=seed))
])