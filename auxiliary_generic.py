import datetime
import holidays
import pandas as pd
import numpy as np
import auxiliary_configuration as auxConfig
import sqlite3

def resampleDataFrame(df_original, configuration):
    period = configuration['resampleFrequency']

    df_original['Date'] = pd.to_datetime(df_original['Date'],utc=True)
        
    df_original.index = pd.DatetimeIndex(data=df_original['Date'])
    df_original = df_original.drop(['Date'], axis=1)
    
    df_resampled = df_original.resample(period).mean()
    
    lenDfOrig = len(df_original.index)
    lenDfResamp = len(df_resampled.index)
    if lenDfOrig < lenDfResamp:
        df_resampled = df_original.resample(period).ffill()
    else:
        df_resampled = df_original.resample(period).mean()
        
    return df_resampled

def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False

def lagged_predictors_pd(df, col_name):
    # tseries = df[col_name].to_numpy()
    # tseries = np.diff(tseries, n=48)
    # tseries = tseries[~np.isnan(tseries)]
    # d = min(d, len(tseries))  # to avoid errors in case of shorter series
    # PACF = pacf(tseries, nlags=d)
    # Lags = np.argwhere(abs(PACF) > thres) - 1
    # starting_point = operational_lag
    # Lags = Lags[Lags > starting_point]
    # Lags = list(range(17,17+48))+list(range(17+289,17+336))+list(range(17+1393,17+1440))+list(range(17+17473,17+17520))
    # Lags = list(range(17,17+48)) # This is for the former paper
    Lags = list(range(17,17+48*7))
    
    
    name_list = []
    for lag in Lags:
        temp_name = col_name + '_l_' + str(lag)
        df[temp_name] = df[col_name].shift(lag)
        name_list.append(temp_name)
    return df, name_list

def treatNAN(df, configuration):
    # to do
    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='all')
    df = df.dropna(axis=0, how='any')
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
    return df

def fillLoadUseOfficial(df):
    emptyLoad = df[df['Load'].isnull() == True]
    df['Load'].loc[emptyLoad.index] = df['ForecastedLoad'].loc[emptyLoad.index]
        

def encodeTimeStamp(df,lags):
    # using the lags to get encoders in the forecasting day
    dayofweek = df.index.dayofweek
    dayofmonth = np.array(list(map(lambda x: x.month,df.index)))
    dayofyear = df.index.dayofyear

    df['dayofweek_sin'] = np.sin(2 * np.pi * dayofweek / 6.0)
    df['dayofweek_cos'] = np.cos(2 * np.pi * dayofweek / 6.0)
    
    df['dayofmonth_sin'] = np.sin(2 * np.pi * dayofmonth / 30.0)
    df['dayofmonth_cos'] = np.cos(2 * np.pi * dayofmonth / 30.0)
    
    df['dayofyear_sin'] = np.sin(2 * np.pi * dayofyear / 365.0)
    df['dayofyear_cos'] = np.cos(2 * np.pi * dayofyear / 365.0)

    newcolumns = ['dayofweek_sin','dayofweek_cos','dayofmonth_sin','dayofmonth_cos','dayofyear_sin','dayofyear_cos']
    
    for col in newcolumns:
        df[col] = df[col].shift(-lags)
    return df, newcolumns

def getClimatolygy_dayOfyear(idx,df,lags):
    '''The df here must be the original time series from 2016'''
    clima = np.zeros(shape=(48,4))
    if idx.month>=6:
        flag = idx.year-2016
    else:
        flag = idx.year-2016-1
    for i in range(1,flag+1):
        '''considering we have 5-year data'''
        newIdx = idx - pd.DateOffset(years=i)
        startIdx,endIdx = newIdx + pd.DateOffset(minutes=30*32), newIdx + pd.DateOffset(minutes=30*(32+47))
        dateRange = pd.date_range(startIdx,endIdx,freq='0.5H')
        if set(dateRange).issubset(set(df.index)) == False:
            clima[:,i-1] = np.zeros((48,))
        else:
            clima[:,i-1] = df['Load'].loc[dateRange]
        
        # while set(dateRange).issubset(set(df.index)) == False:
        #     newIdx = newIdx - pd.DateOffset(days=1)
        #     startIdx,endIdx = newIdx + pd.DateOffset(minutes=30*32), newIdx + pd.DateOffset(minutes=30*(32+47))
        #     dateRange = pd.date_range(startIdx,endIdx,freq='0.5H')
                       
    clima = clima.mean(axis=1)
    return clima
       
def addClimatology(YXtable,df,lags):
    YXtable = YXtable[YXtable.index>'2017-06-02']
    '''This df is the YXtable for training'''
    climaFeat = np.zeros(shape=(len(YXtable),48))
    for i in range(len(YXtable)):
        clima = getClimatolygy_dayOfyear(YXtable.index[i], df, lags)
        if np.all((clima==0)):
            temp = np.empty((1,48))
            temp[:] = np.nan
            climaFeat[i,:] = temp
        else:
            climaFeat[i,:] = clima
    newcolumns = ['Clima-H'+str(i) for i in range(1,49)]
    climaFeatDf = pd.DataFrame(climaFeat,columns=newcolumns,index=YXtable.index)
    YXtable = YXtable.merge(climaFeatDf,left_on='Date', right_on='Date', how='left')
    return YXtable,newcolumns

def getClimatology_hourOfday(idx,df,lags):    
    new_idx = idx-pd.DateOffset(days=1)
    startIdx,endIdx = new_idx-pd.DateOffset(days=7), new_idx
    dateRange = pd.date_range(startIdx,endIdx,freq='0.5H')
    week_mean = df['Load'][df.index.isin(dateRange)].mean()
    
    new_idx = idx-pd.DateOffset(days=1)
    startIdx,endIdx = new_idx-pd.DateOffset(days=30), new_idx
    dateRange = pd.date_range(startIdx,endIdx,freq='0.5H')
    month_mean = df['Load'][df.index.isin(dateRange)].mean()
    
    new_idx = idx-pd.DateOffset(days=1)
    startIdx,endIdx = new_idx-pd.DateOffset(days=365), new_idx
    dateRange = pd.date_range(startIdx,endIdx,freq='0.5H')
    year_mean = df['Load'][df.index.isin(dateRange)].mean()
    
    return np.array([week_mean,month_mean,year_mean])


    
    
    
    

    



def get_UK_holidays_dummies(lags):
    dates = pd.date_range(start='2016-06-01',end='2021-05-31').date
    holidayAll = []
    holidayDict = dict()
    years = list(range(2016,2022))
    for year in years:
        for item in holidays.UnitedKingdom(years = year).items():
            holidayDict[item[0]] = item[1]
    for i in range(len(dates)):
        if dates[i] in holidayDict.keys():
            holidayAll.append(holidayDict[dates[i]])
        else:
            holidayAll.append('Normal')
    
    holidayDf = pd.DataFrame()
    holidayDf['Date'],holidayDf['holiday'] = dates,holidayAll
    holidayDf.set_index('Date')

    df = pd.get_dummies(holidayDf['holiday'])
    df['Date'] = dates
    df.set_index('Date')
    
    df = resampleDataFrame(df, configuration)
    df = df.dropna(axis=0)
    
    return df.shift(-lags)
    
def get_weekends(df,lags):
    temp = df.copy()
    temp['Date'] = pd.to_datetime(temp.index)
    temp['isWeekend'] = temp['Date'].dt.weekday >= 5
    week = pd.get_dummies(temp['isWeekend'])
    week.columns = ['isWeekendFalse','isWeekendTrue']
    del temp
    return week.shift(-lags)
    
def get_temperature(lags,NIE=False):
    #NIE is the indicator of North Ireland
    if NIE==False:
        requestDict = {
                'periodString': 'max',
                'db_connector': configuration['db_connector_address']
                }
        requestDict['tableName'] = 'UKtemperature'
        df = readDataFromDB(requestDict)
    else:
        df = pd.read_csv('./Dataset/Belfast_preprocessed.csv')
        
    df = df.drop_duplicates(subset=['Date'])
    df = resampleDataFrame(df, configuration)
    df = df.dropna(axis=0)
    
    return df.shift(-lags)

def add_rolling_mean(df,lags):
    # add rolling mean for week ahead, month ahead and year ahead
    temp = df.copy()

    if lags == 48:
        text_df = df

    elif lags == 48*7:
        roll_week = temp.rolling(48*7)
        rollDf_week = roll_week.mean()
        rollDf_week.columns = [name+'_mean_week' for name in list(rollDf_week.columns)]       
        text_df = pd.concat([df,rollDf_week],axis=1)

    elif lags == 48*30:
        roll_month = temp.rolling(48*30)
        rollDf_month = roll_month.mean()
        rollDf_month.columns = [name+'_mean_month' for name in list(rollDf_month.columns)]       
        
        text_df = pd.concat([df,rollDf_month],axis=1)

    elif lags == 48*365:
        roll_year = temp.rolling(48*365)
        rollDf_year = roll_year.mean()
        rollDf_year.columns = [name+'_mean_year' for name in list(rollDf_year.columns)]       
        
        text_df = pd.concat([df,rollDf_year],axis=1)

    text_df = text_df.dropna()
    return text_df

configuration = auxConfig.load_forecasting_NLP()

def createXYtables4training(configuration,lags,NIE=False):
    try:
        yColumns = []
        xColumns = []
            
        columnName = 'Load'
        
        if NIE == False:
            requestDict = {
                'periodString': 'max',
                'db_connector': configuration['db_connector_address']
                }
            
            paramTarget = configuration['target']
            requestDict['tableName'] = paramTarget
    
            df = readDataFromDB(requestDict)
        else:
            df = pd.read_csv('./Dataset/NIEload.csv')
            df.columns = ['Date','ForecastedLoad','Load']
        
        df = df.drop_duplicates(subset=['Date'])
        df = resampleDataFrame(df, configuration)

        # Outlier filtering removed - data already preprocessed
        
        # df = df.interpolate()
        
        ForecastedLoad = df[['ForecastedLoad']]
        df = add_rolling_mean(df[['Load']],lags)
        df = df.merge(ForecastedLoad,left_on='Date', right_on='Date', how='left')
        
        YXtable = df.copy()
        
        for h in configuration['horizonts'][str(lags)]:
            newColumnName = columnName + '_h+' + str(h)+ '_'
            YXtable.insert(0, newColumnName, YXtable[columnName].shift(-h))
            yColumns.append(newColumnName)
        
        #using the past day from 0h to 23h30 as the lags
        YXtable, newcolumns = lagged_predictors_pd(YXtable, columnName)
        xColumns.extend(newcolumns)
        
        YXtable, newcolumns = encodeTimeStamp(YXtable,lags)
        xColumns.extend(newcolumns)
        
        YXtable = YXtable.dropna()
        
        YXtableDict,FeatDict = dict(),dict()
        YXtableDict['benchmark'] = YXtable['ForecastedLoad']
    
        # in the paper, keep the time at 8;00
        YXtable = YXtable[YXtable.index.time == datetime.time(8,0)]
        
        FeatDict['Load'] = YXtable.columns.copy()
        
        # add holidays
        holidays = get_UK_holidays_dummies(lags)
        FeatDict['Hol'] = holidays.columns
        holidaysTable = YXtable.merge(holidays,left_on='Date', right_on='Date', how='left')
        
        # add if weekend, if for UK data, the second param is lags; if NIE data, it's 1
        weekend = get_weekends(YXtable,lags)
        FeatDict['week'] = weekend.columns
        weekendTable = YXtable.merge(weekend,left_on='Date', right_on='Date', how='left')
        del weekend
        
        # add temperatures (use Belfast temps when NIE=True)
        temperature = get_temperature(lags, NIE=NIE)
        FeatDict['Temp'] = temperature.columns
        tempTable = YXtable.merge(temperature,left_on='Date', right_on='Date', how='left')
               
        # add both weekends and temperatures
        week_temp_table = weekendTable.merge(temperature,left_on='Date', right_on='Date', how='left')
        
        # add weekends, holidays, temperature
        week_hol_temp_table = week_temp_table.merge(holidays,left_on='Date', right_on='Date', how='left')
        del temperature,holidays
        
        YXtable = YXtable.dropna()
        holidaysTable = holidaysTable.dropna()
        tempTable = tempTable.dropna()
        week_temp_table = week_temp_table.dropna()
        weekendTable = weekendTable.dropna()
        week_hol_temp_table = week_hol_temp_table.dropna()
        
        YXtableDict['YXtable_NoText_Y'] = YXtable[yColumns]
        YXtableDict['YXtable_holiday_Y'] = holidaysTable[yColumns]
        YXtableDict['YXtable_temperature_Y'] = tempTable[yColumns]
        YXtableDict['YXtable_week_temp_Y'] = week_temp_table[yColumns]
        YXtableDict['YXtable_weekend_Y'] = weekendTable[yColumns]
        YXtableDict['YXtable_week_hol_temp_Y'] = week_hol_temp_table[yColumns]
        
        
        YXtableDict['YXtable_NoText_X'] = YXtable.drop(yColumns+['ForecastedLoad'], axis=1)
        YXtableDict['YXtable_holiday_X'] = holidaysTable.drop(yColumns+['ForecastedLoad'], axis=1)
        YXtableDict['YXtable_temperature_X'] = tempTable.drop(yColumns+['ForecastedLoad'], axis=1)
        YXtableDict['YXtable_week_temp_X'] = week_temp_table.drop(yColumns+['ForecastedLoad'], axis=1)
        YXtableDict['YXtable_weekend_X'] = weekendTable.drop(yColumns+['ForecastedLoad'], axis=1)
        YXtableDict['YXtable_week_hol_temp_X'] = week_hol_temp_table.drop(yColumns+['ForecastedLoad'], axis=1)
        
        del holidaysTable,tempTable,weekendTable,week_temp_table,week_hol_temp_table
    except Exception as e:
        print(e)
        
    return YXtableDict,FeatDict




# calculate errors
def calculate_error(Y,col1,col2):
    #Y with y the true valus and yhat the forecast
    Y['horizon'] = [dt.date() for dt in Y.index]
    
    Y['error'] = Y[col1] - Y[col2]
    Y['errorSquare'] = Y['error']**2
    Y['absError'] = Y['error'].abs()
    Y['smapeError'] = Y['absError']/((Y[col1] + Y[col2])/2)
    
    Error = pd.DataFrame(columns=['rmse', 'mae', 'smape'])
    Error['rmse'] = Y.groupby('horizon').mean()['errorSquare']**0.5
    Error['mae'] = Y.groupby('horizon').mean()['absError']
    Error['smape'] = Y.groupby('horizon').mean()['smapeError']
    return Error

def scale_data(cols,scaler,data):
    if type(data) == pd.core.series.Series:
        data = data.to_frame()
    index_data = data.index
    data = scaler.transform(data)
    data = pd.DataFrame(data,columns=cols).set_index(index_data)
    return data

def expand_data(data):
    fore_df = pd.DataFrame()
    for i in range(len(data)):
        temp = data.iloc[i,:]
        temp_index = [data.index[i] + datetime.timedelta(minutes=30*j) for j in range(32,32+48)]
        temp = temp.to_frame()
        temp.columns = ['temp']
        temp = temp.reset_index(drop=True)
        temp['Date'] = temp_index
        temp = temp.set_index('Date')
        fore_df = pd.concat([fore_df,temp])
    return fore_df







def readDataFromDB(requestDict):
    tableName = requestDict['tableName']
    db_connector = requestDict['db_connector']
    conn = sqlite3.connect(db_connector)
    
    sqlString = ' SELECT * FROM ' + tableName
    df_temp = pd.read_sql_query(sqlString, conn)
    conn.close()

    return df_temp
