import numpy as np
import json

def load_forecasting_NLP():
    configuration = {'configuration_name': 'Test_for_electric_load_forecasting'}
    configuration['model_name'] = 'ExTr01'
    configuration['description'] = 'extra tree regressor based on Akylas baseline'
    configuration['traintestsize'] = 0.5
    configuration['horizonts'] = {str(48):list(range(32,80)),
                                  str(int(48*7)):list(range(32,32+48*7)),
                                  str(int(48*30)):list(range(32,32+48*30)),
                                  str(int(48*365)):list(range(32,32+48*365))}
    
    configuration['quantiles'] = list(np.arange(0.01, 0.09, 0.01))

    # paths
    configuration['db_connector_address'] = 'Dataset/LoadSeriesForecasting-NLP.db'
    configuration['dbOut_connector_address'] = 'output/Forecasting_NLP_Output.db'
    configuration['remote_datasource'] = 'yFinance' #or entsoe
    configuration['pathInput'] = 'input/'
    configuration['pathOutput'] = 'output/'
    configuration['pathModels'] = 'models/'
    configuration['pathEvaluation'] = 'evaluation/'
    configuration['pathFeatures'] = 'features/'

    # actions
    configuration['download'] = 'no'
    configuration['sleep'] = 2
    configuration['train'] = 'yes'
    configuration['forecast'] = 'yes'
    configuration['plot'] = 'yes'
    configuration['evaluate'] = 'yes'
    configuration['featureselection'] = 'yes'
    configuration['crossvalidation'] = 'no'
    configuration['saveXYtable'] = 'no'
    configuration['savetestpreds'] = 'yes'

    configuration['resampleFrequency'] = '0.5H' # from: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    configuration['date2Forecast'] = ['2011-07-16'] # not yet implemented

    # models metaparameters
    configuration['n_lagsMax'] = 48
    configuration['n_estimators'] = 100
    configuration['min_features_to_select'] = 10
    configuration['n_scenarios'] = configuration['n_estimators']

    # preprocessing, features selection, tuning
    configuration['tuning_parameters'] = {
        'max_features':[0.65, 0.7, 0.75, 0.8, 0.85],
        'n_estimators':[100, 500],
        'bootstrap': [False]
    }
    configuration['score'] = 'neg_mean_absolute_error' # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    configuration['nan_approach'] = 'impute'  # drop #impute #no

    # parameters for evaluation
    configuration['eval_sharpness_interval'] = [0.1, 0.9] #must be in the quantiles calculated
    configuration['eval_QS_min'] = 0
    configuration['eval_QS_max'] = 0.2

    # postprocessing
    configuration['parameters'] = []
    configuration['target'] = 'UKLoadForeAll'
    configuration['NLP_features_table'] = ['title_features_count', 'title_WordFreqDf','title_features_senti',
                                            'title_topicDistributionDf_grouped',
                                            'title_features_Glove_grouped', 
                                            'des_features_count', 'des_WordFreqDf', 'des_features_senti',
                                            'des_topicDistributionDf_grouped', 
                                            'des_features_Glove_grouped', 
                                            'body_features_count', 'body_WordFreqDf', 'body_features_senti',
                                            'body_topicDistributionDf_grouped', 
                                            'body_features_Glove_grouped',
                                            'text_features_bert']
    
    configuration['useful_table'] = ['title_WordFreqDf','des_WordFreqDf','body_WordFreqDf',
                                     'body_features_senti','body_topicDistributionDf_grouped',
                                     'body_features_Glove_grouped',
                                     'text_features_bert']  
    
    # configuration['NLP_features_table'] = ['UKtemperature']
    configuration['all_text_features_table'] = ['title','des','body','allText']
    configuration['quantiles'] = np.around(np.arange(.1, 1, .1),decimals=1)
    return configuration














