import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import auxiliary_generic as auxGen
from sklearn.metrics import mean_pinball_loss
import os
import shutil
from datetime import datetime
from bs4 import BeautifulSoup as bs
import os
import re
import random


def calc_crps(obsAndPreds, configuration):
    quantiles = configuration['quantiles']
    col = 'Load_observations'
    dt = 'Date'
    fl = 'ForecastedLoad'
    quantColumn = [column for column in obsAndPreds.columns if 'Load_Q' in column]
    dfTemp = obsAndPreds.loc[:, quantColumn]
    dfObs = obsAndPreds.loc[:, obsAndPreds.columns == col]
    y = dfTemp.sub(dfObs[col], axis=0).pow(2).to_numpy()
    crps = np.trapz(y, quantiles)
    return crps

def calc_sharpness(obsAndPreds, configuration):
    sharpnessInterval = configuration['eval_sharpness_interval']
    col = 'Load'
    [quantileLow, quantileHigh] = [sharpnessInterval[0], sharpnessInterval[1]]
    quantileLow = col + '_Q' + str(quantileLow) + '_'
    quantileHigh = col + '_Q' + str(quantileHigh) + '_'
    quantileLow = obsAndPreds[quantileLow]
    quantileHigh = obsAndPreds[quantileHigh]
    sharpness = quantileHigh - quantileLow
    return sharpness

def calc_reliability(obsAndPreds, configuration):
    quantiles = configuration['quantiles']
    horizonts = obsAndPreds['horizon'].unique()
    colNames = ['horizon']
    colNames.extend(quantiles)
    dfReliability = pd.DataFrame(columns=colNames)#, index=horizonts)
    for h in horizonts:
        y_true = obsAndPreds.loc[obsAndPreds['horizon'] == h]['Load_observations']
        reliabilityList = [h]
        for q in quantiles:
            y_pred = obsAndPreds.loc[obsAndPreds['horizon'] == h]['Load_Q' + str(q) + '_']
            x = np.mean(y_true<y_pred)
            reliabilityList.append(x)

        dfReliability.loc[len(dfReliability)] = reliabilityList
    dfReliability.set_index('horizon', inplace=True)

    return dfReliability

def calc_pinball(obsAndPreds, configuration):
    quantiles = configuration['quantiles']
    horizonts = obsAndPreds['horizon'].unique()
    colNames = ['horizon']
    colNames.extend(quantiles)
    dfPinball = pd.DataFrame(columns=colNames)#, index=horizonts)
    for h in horizonts:
        y_true = obsAndPreds.loc[obsAndPreds['horizon'] == h]['Load_observations']
        pinballList = [h]
        for q in quantiles:
            y_pred = obsAndPreds.loc[obsAndPreds['horizon'] == h]['Load_Q' + str(q) + '_']
            pinballList.append(mean_pinball_loss(y_true, y_pred, alpha=q))

        dfPinball.loc[len(dfPinball)] = pinballList
    dfPinball.set_index('horizon', inplace=True)

    return dfPinball

def calc_scenariosScore(obsAndPreds, configuration):
    col = 'Load_observations'
    dfScen = obsAndPreds.loc[:, obsAndPreds.columns != col]
    dfObs = obsAndPreds.loc[:, obsAndPreds.columns == col]

    dfTemp = dfScen.sub(dfObs[col], axis=0).abs()
    uniqueIssuedTimestamps = obsAndPreds.index.get_level_values(0).unique()

    colsNames = ['issued-timestamp']
    colsNames.extend(range(configuration['n_scenarios']))
    dfScenScore = pd.DataFrame(columns=colsNames)
    for issue in uniqueIssuedTimestamps:
        x = dfTemp.iloc[dfTemp.index.get_level_values(0) == issue]
        x = x.sum(axis=0).to_list()
        x.sort()
        listScen = [issue]
        listScen.extend(x)
        dfScenScore.loc[len(dfScenScore)] = listScen
    dfScenScore = dfScenScore.mean(axis=0)
    dfScenScore = dfScenScore.to_frame(name = 'scenScore')
    return dfScenScore

def evaluate_deterministic(Y, configuration):
    stringPred = "Load_deterministic"
    stringObs = 'Load_observations'
    # Y['horizon'] = Y.index.get_level_values(1) - Y.index.get_level_values(0)
    dateList = Y['Date'].tolist()
    Y['horizon'] = [dt.date() for dt in dateList]
    
    Y['WEerror'] = Y[stringPred] - Y[stringObs]
    Y['WEerrorSquare'] = Y['WEerror']**2
    Y['WEabsError'] = Y['WEerror'].abs()
    Y['WEsmapeError'] = Y['WEabsError']/((Y[stringPred] + Y[stringObs])/2)
    
    Y['OFIIerror'] = Y['ForecastedLoad'] - Y[stringObs]
    Y['OFIIerrorSquare'] = Y['OFIIerror']**2
    Y['OFIIabsError'] = Y['OFIIerror'].abs()
    Y['OFIIsmapeError'] = Y['OFIIabsError']/((Y['ForecastedLoad'] + Y[stringObs])/2)

    dfEvaluationDeterministic = pd.DataFrame(columns=['WErmse', 'WEmae', 'WEsmape',
                                                      'OFIIrmse','OFIImae','OFIIsmape'])
    dfEvaluationDeterministic['WErmse'] = Y.groupby('horizon').mean()['WEerrorSquare']**0.5
    dfEvaluationDeterministic['WEmae'] = Y.groupby('horizon').mean()['WEabsError']
    dfEvaluationDeterministic['WEsmape'] = Y.groupby('horizon').mean()['WEsmapeError']

    dfEvaluationDeterministic['OFIIrmse'] = Y.groupby('horizon').mean()['OFIIerrorSquare']**0.5
    dfEvaluationDeterministic['OFIImae'] = Y.groupby('horizon').mean()['OFIIabsError']
    dfEvaluationDeterministic['OFIIsmape'] = Y.groupby('horizon').mean()['OFIIsmapeError']

    return dfEvaluationDeterministic

def evaluate_quantiles(Y,configuration):
    sharpness = calc_sharpness(Y, configuration)
    crps = calc_crps(Y, configuration)  
    dateList = Y['Date'].tolist()
    Y['horizon'] = [dt.date() for dt in dateList]
    # Y['horizon'] = Y.index.get_level_values(1) - Y.index.get_level_values(0)
    
    Y['sharpness'] = sharpness
    Y['crps'] = crps
    dfEvaluationQuantiles_byHorizon = pd.DataFrame()
    dfEvaluationQuantiles_byHorizon['crps'] = Y.groupby('horizon').mean()['crps']
    dfEvaluationQuantiles_byHorizon['sharpness'] = Y.groupby('horizon').mean()['sharpness']

    dfPinball = calc_pinball(Y, configuration)
    dfReliability = calc_reliability(Y, configuration)

    return dfEvaluationQuantiles_byHorizon, dfPinball, dfReliability

def evaluate_scenarios(Y,configuration):
    stringPred = "Load_deterministic"
    stringObs = 'Load_observations'
    Y['horizon'] = Y.index.get_level_values(1) - Y.index.get_level_values(0)
    Y['error'] = Y[stringPred] - Y[stringObs]
    Y['absError'] = Y['error'].abs()

    dfEvaluationDeterministic = pd.DataFrame(columns=['bias', 'mae', 'std'])
    dfEvaluationDeterministic['bias'] = Y.groupby('horizon').mean()['error']
    dfEvaluationDeterministic['mae'] = Y.groupby('horizon').mean()['absError']
    dfEvaluationDeterministic['std'] = Y.groupby('horizon').std()['error']
    return dfEvaluationDeterministic


def evaluate_scenarios_toCheck(Y,configuration):
    dfEvaluationScenarios = calc_scenariosScore(Y,configuration)
    # add energy score


    return dfEvaluationScenarios

def plotPredictions(dfDet, dfQua, dfSce, configuration, namePic):
    # check if observation column is in dframes, if not, it's because the prediction is in the future
    modelName = auxGen.createModelName(configuration)
    pathName = configuration['pathEvaluation'] + modelName + '/'
    issued_timestamp = dfDet.index.get_level_values(0).min()
    from_timestamp = dfDet.index.get_level_values(1).min()
    to_timestamp = dfDet.index.get_level_values(1).max()

    title = 'Predictions for Load'
    title = title + '\n issued on ' + issued_timestamp.isoformat() + '\n from ' + from_timestamp.isoformat() + ' to ' + to_timestamp.isoformat()

    plt.figure()
    plt.title(title)

    # plot deterministic forecasts
    x = dfDet.index.get_level_values(1)
    y = dfDet['Load_deterministic']
    plt.plot(x, y, color='black', linestyle='dashed')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # check if observations are available, if so, plot them
    col = 'Load_observations'
    if col in dfDet.columns:
        y = dfDet[col]
        plt.plot(x, y, color='black', marker='+', linestyle="None")

    # plot quantiles intervals
    quantiles = configuration['quantiles']
    for i in range(0,len(quantiles)-1):
        qTop = quantiles[i+1]
        qDown = quantiles[i]
        alpha = 0.5-max(abs(qTop-0.5),abs(qDown-0.5))
        y2 = dfQua['Load_Q' + str(qTop) + '_']
        y1 = dfQua['Load_Q' + str(qDown) + '_']
        plt.fill_between(x, y1=y1, y2=y2, where=None, interpolate=True, alpha=alpha, facecolor='b', step='mid')

    # plot scenarios
    for i in range(0,configuration['n_scenarios']):
        y = dfSce['Load_S' + str(i) + '_']
        plt.plot(x, y, color='black', linewidth=0.1)
    plt.savefig(pathName + namePic + '.jpg')


def createEvalutionReport_SVR_Ada(dfEvDet, configuration, tableName, MLname):
    # to do: add y_test and x_test as input and create reports for target and features
    # to do: add features importance

    # create folder
    pathName = configuration['pathEvaluation'] + tableName+'_'+MLname + '/'
    figurename = tableName +'_'+MLname

    # plot deterministic evaluation
    plot2 = dfEvDet.plot(lw=2, marker='.', markersize=10, title= 'Deterministic error evaluation' + figurename) #colormap='tab10', 'Accent'....
    plot2.set_xlabel("Horizon")
    plot2.set_ylabel("Values")
    fig2 = plot2.get_figure()
    fig2.tight_layout()
    fig2.savefig(pathName + "Deterministic.jpg")
    dfEvDet.to_csv(pathName + "evaluationDeterministic.csv")

def createEvalutionReport_ExtraTrees(dfEvDet, dfEvQ, dfPinball, dfReliab, configuration,dfFeatImp,tableName, MLname):
    # to do: add y_test and x_test as input and create reports for target and features
    # to do: add features importance

    # create folder
    pathName = configuration['pathEvaluation'] + tableName+'_'+MLname + '/'
    figurename = tableName +'_'+MLname

    # plot deterministic evaluation
    plot2 = dfEvDet.plot(lw=2, marker='.', markersize=10, title= 'Deterministic error evaluation' + figurename) #colormap='tab10', 'Accent'....
    plot2.set_xlabel("Horizon")
    plot2.set_ylabel("Values")
    fig2 = plot2.get_figure()
    fig2.tight_layout()
    fig2.savefig(pathName + "Deterministic.jpg")
    dfEvDet.to_csv(pathName + "evaluationDeterministic.csv")

    # plot quantile evaluation
    plot3 = dfEvQ.plot(lw=2, marker='.', markersize=10,title= 'Quantile error evaluation' + figurename, secondary_y=['crps']) #colormap='tab10', 'Accent'....
    plot3.set_xlabel("Horizon")
    plot3.set_ylabel("Values")
    fig3 = plot3.get_figure()
    fig3.tight_layout()
    fig3.savefig(pathName + "Quantile.jpg")
    dfEvQ.to_csv(pathName + "evaluationQuantiles.csv")

    # plot pinball
    dfPinball = dfPinball.transpose()
    plot4 = dfPinball.plot(lw=2, title= 'Pinball loss' + figurename)  # colormap='tab10', 'Accent'....
    plot4.set_xlabel("Quantiles")
    plot4.set_ylabel("Value")
    fig4 = plot4.get_figure()
    fig4.tight_layout()
    fig4.savefig(pathName + "Pinball.jpg")
    dfPinball.to_csv(pathName + "evaluationPinball.csv")

    # plot reliability
    dfReliab = dfReliab.transpose()
    plot5 = dfReliab.plot(lw=2, title='Reliability diagram'+ figurename) #colormap='tab10', 'Accent'....
    plot5.set_xlabel("Predicted")
    plot5.set_ylabel("Observed")
    fig5 = plot5.get_figure()
    fig5.tight_layout()
    fig5.savefig(pathName + "Reliability.jpg")
    dfReliab.to_csv(pathName + "evaluationReliability.csv")


    # plot feature importance
    plot7 = dfFeatImp.plot.barh(x='name', y='value')
    fig7 = plot7.get_figure()
    fig7.tight_layout()
    fig7.savefig(pathName + "featureImportance.jpg")
    dfFeatImp.to_csv(pathName + "featureImportance.csv")



