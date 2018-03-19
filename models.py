#Read in packages needed to do the modeling
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor #note: Xcode doesnt support OpenMP multi-threading, which XGBoost uses. To fix this you run 'brew install gcc@5' in the terminal to fix.
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error, explained_variance_score

seed = 4



def split_data(df, target_col):
    no_missing_target = df.loc[df[target].isna()]
    target = no_missing_target[target_col]
    features = no_missing_target.drop([target_col], axis =1)
    feature_model, feature_holdout, target_model, target_holdout = train_test_split(features, target, test_size=0.1, random_state=seed)
    return(feature_model, feature_holdout, target_model, target_holdout)


from sklearn.model_selection import cross_val_score
def cv_model(features, target, model, kfolds = 5):
    
    #features.columns = [c.replace(' ', '_') for c in features.columns]
    scores = cross_val_score(model, features, target, cv=kfolds)
    
    feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size, random_state=seed)
    
    #build the model
    model = XGBRegressor(max_depth=max_tree_depth)
    model.fit(feature_train, target_train)
    return(model)


    #make prediction
    target_pred = model.predict(feature_test)
    #get model scores
    r2 = r2_score(target_test, target_pred)
    mean_sq_error = mean_squared_error(target_test, target_pred)
    median_abs_error = median_absolute_error(target_test, target_pred)
    expl_var_score = explained_variance_score(target_test, target_pred)
    
    score_df  =pd.DataFrame({"R^2 Score": r2,
                             "Mean Squared Error": mean_sq_error,
                             "Median Absolute Error": median_abs_error},
                             index =["Model"]).transpose()
    return(score_df)