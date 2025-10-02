import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV


def featureSelction(X, Y, min_features_to_select):
    estimator = ExtraTreesRegressor(n_estimators=10)
    selector = RFECV(estimator, min_features_to_select=min_features_to_select)
    selector = selector.fit(X, Y)
    a = selector.support_
    Xnew = X.loc[:,a]
    return Xnew

def tuningCV(X, Y, n_estimators, tuning_parameters, score):
    #https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
    estimator = ExtraTreesRegressor(n_estimators=n_estimators)
    clf = GridSearchCV(estimator=estimator, param_grid=tuning_parameters, scoring=score)
    clf.fit(X, Y)
    bestEstimator = clf.best_estimator_
    return bestEstimator


def fit(X_train, Y_train, configuration):
    # to do:
    # test https://github.com/cerlymarco/shap-hypetune
    # for feature selection and tuning

    if configuration['featureselection'] == 'yes':
        X_train = featureSelction(X_train, Y_train, configuration['min_features_to_select'])
    else:
        X_train = X_train

    if configuration['crossvalidation'] == 'yes':
        estimator = tuningCV(X_train, Y_train, configuration['n_estimators'],
                             configuration['tuning_parameters'], configuration['score'])
    else:
        estimator = ExtraTreesRegressor(n_estimators=int(configuration['n_estimators']))
        estimator = estimator.fit(X_train, Y_train)

    setattr(estimator, 'names_inputs', list(X_train.columns))
    setattr(estimator, 'names_outputs', list(Y_train.columns))
    return estimator

def predict(est, X, quantiles, horizonts, resampleFrequency, target):
    columnsY = est.names_outputs
    columnsX = est.names_inputs
    d = [s + 'deterministic' for s in columnsY]
    #X = X[X.columns.intersection(columnsX)]
    X = X.loc[:,columnsX]
    Y = est.predict(X)
    dfOut = pd.DataFrame(Y, columns=d, index=X.index)

    # get prediction from each tree
    for i in range(len(est.estimators_)):
        columnsYScenPred = [s + 'S' + str(i) + '_' for s in columnsY]
        s = pd.DataFrame(est.estimators_[i].predict(X.values), columns=columnsYScenPred, index=X.index)
        s.index.name = 'Date'
        dfOut = dfOut.merge(s, left_on='Date', right_on='Date', how='inner')

    # get quantiles for each horizon
    for h in horizonts:
        strCol = target + '_h+' + str(h) + '_S'
        dfTemp = dfOut[[col for col in dfOut.columns if strCol in col]]
        q = dfTemp.quantile(q=quantiles, axis=1).T
        strCol = target + '_h+' + str(h) + '_'
        q.columns = [strCol + 'Q' + s + '_' for s in [str(x) for x in q]]
        q.index.name = 'Date'
        dfOut = dfOut.merge(q, left_on='Date', right_on='Date', how='inner')

    # reshuffle each horizon under the precedent
    issuedTimestamp = dfOut.index
    dfOutByH = pd.DataFrame()
    for h in horizonts:
        forecastTimestamp = dfOut.index
        forecastTimestamp = forecastTimestamp.shift(h, freq=resampleFrequency)
        strCol = target + '_h+' + str(h) + '_'
        dfTemp = dfOut[[col for col in dfOut.columns if strCol in col]]
        dfTemp['issued-timestamp'] = issuedTimestamp
        dfTemp['forecast-timestamp'] = forecastTimestamp
        dfTemp.set_index(['issued-timestamp', 'forecast-timestamp'], inplace=True)
        newCols = [s.replace('_h+' + str(h) + '_', '_') for s in dfTemp.columns]
        dfTemp.columns = newCols
        dfOutByH = pd.concat([dfOutByH, dfTemp], ignore_index=False, sort=False)

    # create df for deterministic forecasts
    strCol = target + '_deterministic'
    dfDeterministic = dfOutByH[[col for col in dfOutByH.columns if strCol in col]]

    # create df for quantiles forecasts
    strCol = target + '_Q'
    dfQuantiles = dfOutByH[[col for col in dfOutByH.columns if strCol in col]]

    # create df for scenarios forecasts
    strCol = target + '_S'
    dfScenarios = dfOutByH[[col for col in dfOutByH.columns if strCol in col]]

    return [dfDeterministic, dfQuantiles, dfScenarios]

