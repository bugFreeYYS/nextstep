import pandas as pd 
import numpy as np
import itertools

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import cross_validate

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
import statsmodels



def cleanParamDF(results):
    df = pd.DataFrame.from_dict(results)
    temp = pd.DataFrame(list(df['params']))
    temp['MSE'] = df['mean_test_score'].apply(lambda x: np.sqrt(abs(x)))
    return temp


# can change the print statement to return statement,
# depends on how do we want to put everything together 

def GridSearch(x_train, y_train, x_test, y_test):
    predicted_values = {
    'actual': y_test
    }
    #SARIMA
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p,d,q))
    seasonal_pdq = [(x[0], x[1], x[2], 48) for x in list(itertools.product(p, d, q))]

    dic = dict()
    RMSE = []
    temp_fitted_values = dict()
    df_rows = []    
    tempRMSE = 100000

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            param_row = list(param) + list(param_seasonal)
            mod = sm.tsa.statespace.SARIMAX(y_train,
                                           order=param,
                                           seasonal_order=param_seasonal,
                                           enforce_stationary=False,
                                           enforce_invertibility=False)

            results = mod.fit()

            try:
                rmse = np.sqrt(mean_squared_error(y_train, results.fittedvalues))
                mae = mean_absolute_error(y_train, results.fittedvalues)
                # temp_fitted_values[str(rmse)] = results.fittedvalues[1:]

            except:
                rmse = np.sqrt(mean_squared_error(y_train[1:], results.fittedvalues))
                mae = mean_absolute_error(y_train[1:], results.fittedvalues)   
                # temp_fitted_values[str(rmse)] = results.fittedvalues
                 

            param_row.append(rmse)
            param_row.append(mae)
            if len(param_row) == 9:
                df_rows.append(param_row)        

            dic[str(rmse)] = [param, param_seasonal]
            if rmse < tempRMSE:
                tempRMSE = rmse
                bestmodel = results
            RMSE.append(rmse)

                

    df = pd.DataFrame(df_rows, columns=["p", "d", "q", "seasonal_p", "seasonal_d", "seasonal_q", "seasonality", "RMSE", "MAE"])
    # predicted_values['SARIMA'] = temp_fitted_values[str(min(RMSE))]
    df.to_excel("SARIMA Parameters.xlsx")
    print("SARIMA Best Score: {}".format(min(RMSE)))
    print("SARIMA Best Set of Parameters: {} , {}".format(dic[str(min(RMSE))][0], dic[str(min(RMSE))][1]))

    predicted_values['SARIMA'] = bestmodel.predict(start=480, end=527)
                    
    # ARIMA
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p,d,q))
    dic = dict()
    RMSE = []
    temp_fitted_values = dict()
    df_rows = []
    tempRMSE = 100000
    for param in pdq:
        param_row = list(param)
        mod = statsmodels.tsa.arima_model.ARIMA(y_train,
                                       order=param)

        results = mod.fit()

        try:
            rmse = np.sqrt(mean_squared_error(y_train, results.fittedvalues))
            mae = mean_absolute_error(y_train, results.fittedvalues)
            # temp_fitted_values[str(rmse)] = results.fittedvalues[1:]
        except:
            rmse = np.sqrt(mean_squared_error(y_train[1:], results.fittedvalues))
            mae = mean_absolute_error(y_train[1:], results.fittedvalues)
            # temp_fitted_values[str(rmse)] = results.fittedvalues
        
        param_row.append(rmse)
        param_row.append(mae)
        if len(param_row) == 5:
            df_rows.append(param_row)    

        dic[str(rmse)] = param
        if rmse < tempRMSE:
            tempRMSE = rmse
            bestmodel = results
        RMSE.append(rmse)

    df = pd.DataFrame(df_rows, columns=["p", "d", "q", "RMSE", "MAE"])
    # predicted_values['ARIMA'] = temp_fitted_values[str(min(RMSE))]
    df.to_excel("ARIMA Parameters.xlsx")
    print(" ")
    print("ARIMA Best Score: {}".format(min(RMSE)))
    print("ARIMA Best Set of Parameters: {}".format(dic[str(min(RMSE))]))

    predicted_values['ARIMA'] = bestmodel.predict(start=480, end=527)

                                    
    #AdaBoost
    model = AdaBoostRegressor()

    parameters = {
        'n_estimators': [15, 20, 40],
        'learning_rate': [0.05, 0.07, 0.1],
        'loss': ['linear', 'square', 'exponential']
    }

    grid = GridSearchCV(model,
                        parameters,
                        scoring = 'neg_mean_squared_error',
                        cv = 4, 
                        n_jobs = -1,
                        verbose = True)

    grid.fit(x_train, y_train)
    predicted_values['AdaBoost'] = grid.predict(x_test)
    print('AdaBoost Best Score: {}'.format(np.sqrt(abs(grid.best_score_))))
    print('AdaBoost Best Set of Parameters: {}'.format(grid.best_params_))
    temp_df = cleanParamDF(grid.cv_results_)
    temp_df.to_excel('AdaBoost Parameters.xlsx')
    


    #Random Forest
    model = RandomForestRegressor()

    parameters = {
      'criterion':['mse'],
      'max_depth': [5, 20, 60],
      'min_samples_leaf': [5, 10, 20],
      'max_features': [10, 13, 15],
      'bootstrap': [True, False],
      'n_estimators': [15, 20, 40]

    }

    grid = GridSearchCV(model,
                        parameters,
                        scoring = 'neg_mean_squared_error',
                        cv = 4,
                        n_jobs = -1,
                        verbose=True)

    grid.fit(x_train, y_train)
    predicted_values['Random Forest'] = grid.predict(x_test)
    print('Random Forest Best Score: {}'.format(np.sqrt(abs(grid.best_score_))))
    print('Random Forecast Best Set of Parameters: {}'.format(grid.best_params_))
    temp_df = cleanParamDF(grid.cv_results_)
    temp_df.to_excel('Random Forest Parameters.xlsx')
    

    
    #XGBoost
    model = xgb.XGBRegressor()

    parameters = {#'nthread':[4], #when use hyperthread, xgboost may become slower
      'booster':['gbtree', 'gblinear', 'dart'],
      'learning_rate': [0.05, 0.07, 0.1], #so called `eta` value
      'max_depth': [5, 20, 60],
      'min_child_weight': [10, 20, 30],
      'silent': [1],
      'subsample': [0.7],
      'colsample_bytree': [0.7],
      'n_estimators': [15, 20, 40],
      'tree_method':['auto', 'exact', 'approx', 'hist', 'gpu_hist']
    }

    grid = GridSearchCV(model,
                        parameters,
                        scoring = 'neg_mean_squared_error',
                        cv = 4,
                        n_jobs = -1,
                        verbose=True)

    grid.fit(x_train, y_train)
    predicted_values['XGboost'] = grid.predict(x_test)
    print('XGboost Best Score: {}'.format(np.sqrt(abs(grid.best_score_))))
    print('XGboost Best Set of Parameters: {}'.format(grid.best_params_))
    temp_df = cleanParamDF(grid.cv_results_)
    temp_df.to_excel('XGboost Parameters.xlsx')
    
    for key in predicted_values:
        predicted_values[key] = list(predicted_values[key][:48])
    
    # exporting all predicted values
    outputDF = pd.DataFrame.from_dict(predicted_values)
    outputDF.to_excel('predicted_values.xlsx')

    