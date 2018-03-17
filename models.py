#Read in packages needed to do the modeling
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor #note: Xcode doesnt support OpenMP multi-threading, which XGBoost uses. To fix this you run 'brew install gcc@5' in the terminal to fix.
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error, explained_variance_score

seed = 4



def high_level_cleaning(df, bad_features, target_col):
    model_data = df[df.columns.difference(bad_features)].dropna(axis=0, subset=[target_col])
    no_missing_target = model_data.loc[model_data[target].isna()]
    target = no_missing_target[target_col]
    features = model_data2.drop([target_col], axis =1)
    return(features, target)

#So I realized that I might want to run multiple data sets, I'm just going to make this into a function
def xg_boost_model(df, target_col, bad_features, max_tree_depth, test_size =0.25, seed = 42):
    #remove bad features and drop rows with a null target
    model_data = df[df.columns.difference(bad_features)].dropna(axis=0, subset=[target_col])
    #we also want to remove any targets that have  LTV of 0. LTV=0 doesn't make any sense since we're giving them loans
    model_data2 = model_data.loc[model_data['LTV']>0]
    
    #split dataframe in to target and features
    target = model_data2[target_col]
    features = model_data2.drop([target_col], axis =1)
    #remove whitespace from feature names. Whitespace breaks xgboost for some reason
    features.columns = [c.replace(' ', '_') for c in features.columns]
    
    
    feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size, random_state=seed)
    
    #build the model
    model = XGBRegressor(max_depth=max_tree_depth)
    model.fit(feature_train, target_train)
    
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