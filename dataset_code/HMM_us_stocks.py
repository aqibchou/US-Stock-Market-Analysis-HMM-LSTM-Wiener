import pickle
import numpy as np
from dataset_code.process_us_stock_raw_data import form_us_stock_dataset
from public_tool.handle_us_stock_outliers import handle_us_stock_outliers
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import pandas as pd
from public_tool.combine_us_stock_data_us_stock_data_us_stock_flags import combine_us_stock_data_us_stock_data_us_stock_flags
from public_tool.get_us_stock_index import get_us_stock_index
from public_tool.form_us_stock_model_dataset import form_us_stock_model_dataset


def load_us_stock_single_score():
    # Returns score, feature_name

    temp = pickle.load(open('data/us_stocks/us_stock_solve1_score.csv', 'rb'))
    return temp[0], temp[1]


def form_us_sector_types():
    # Returns a general recorder for various types of multi-factor features, list type, containing lists of category counts, each list is the name of that category

    # Quality factors, describing asset-liability, turnover, operation, profit, cost and expense indicators
    quality_factors = ['AccountsPayablesTDays', 'AccountsPayablesTRate', 'AccountsPayablesTRate', 'ARTDays', 'ARTDays', 'ARTDays', 'BLEV', 'BondsPayableToAsset', 'BondsPayableToAsset', 'CashRateOfSales', 'CashToCurrentLiability', 'CurrentAssetsRatio', 'CurrentRatio', 'DebtEquityRatio', 'DebtEquityRatio', 'DebtsAssetRatio', 'EBITToTOR', 'EquityFixedAssetRatio', 'EquityToAsset', 'EquityTRate', 'FinancialExpenseRate', 'FixAssetRatio', 'FixedAssetsTRate', 'GrossIncomeRatio', 'IntangibleAssetRatio', 'InventoryTDays', 'InventoryTRate', 'LongDebtToAsset', 'LongDebtToWorkingCapital', 'LongTermDebtToAsset', 'MLEV', 'NetProfitRatio', 'NOCFToOperatingNI', 'NonCurrentAssetsRatio', 'NPToTOR', 'OperatingExpenseRate', 'OperatingProfitRatio', 'OperatingProfitToTOR', 'OperCashInToCurrentLiability', 'QuickRatio', 'ROA', 'ROA5', 'ROE', 'ROE5', 'SalesCostRatio', 'SaleServiceCashToOR', 'TaxRatio', 'TotalAssetsTRate', 'TotalProfitCostRatio', 'CFO2EV', 'ACCA', 'DEGM']
    # Describes return and risk
    return_risk_factors = ['CMRA', 'DDNBT', 'DDNCR', 'DDNSR', 'DVRAT', 'HBETA', 'HSIGMA', 'TOBT', 'Skewness', 'BackwardADJ']
    # Describes market value, P/E, P/B
    valuation_factors = ['CTOP', 'CTP5', 'ETOP', 'ETP5', 'LCAP', 'LFLO', 'PB', 'PCF', 'PE', 'PS', 'FY12P', 'SFY12P', 'TA2EV', 'ASSI']
    # Sentiment class, describing psychology, turnover rate, dynamic buying/selling, volume, popularity, willingness, market trend
    sentiment_factors = ['DAVOL10', 'DAVOL20', 'DAVOL5', 'MAWVAD', 'PSY', 'RSI', 'VOL10', 'VOL120', 'VOL20', 'VOL240', 'VOL5', 'VOL60', 'WVAD', 'ADTM', 'ATR14', 'QTR6', 'SBM', 'STM', 'OBV', 'OBV6', 'TVMA20', 'TVMA6', 'TVSTD20', 'TVSTD6', 'VDEA', 'VDIFF', 'VEMA10', 'WEMA12', 'VEMA26', 'VEMA5', 'VMACD', 'VOSC', 'VR', 'VROC12', 'VROC6', 'VSTD10', 'VSTD20', 'ACD6', 'ACD20', 'AR', 'BR', 'ARBR', 'NVI', 'PVI', 'JDQS20', 'KlingerOscillator', 'MoneyFlow20', 'Volatility']
    # Technical indicator class, moving averages, calculation periods, dynamic movement, differences
    technical_indicators = ['MassIndex', 'SwingIndex', 'minusDI', 'plusDI', 'ChaikinVolatility', 'ChaikinOscillator', 'DownRVI', 'BollUp', 'BollDown', 'DHILO', 'EMA10', 'EMA120', 'EMA20', 'EMA5', 'EMA60', 'EA10', 'EA120', 'EA20', 'EA5', 'EA60', 'MFI', 'ILLIQUIDITY', 'MACD', 'KDJ_K', 'KDJ_D', 'KDJ_J', 'UpRVI', 'RVI', 'DBCD', 'ASI', 'EMV12', 'EMV6', 'ADX', 'ADXR', 'MTM', 'MTMMA', 'UOS', 'EMA12', 'EMA26', 'BBI', 'TEMA10', 'Ulcer10', 'Hurst', 'Ulcer5', 'TEMA5', 'CR20', 'Elder', 'DilutedEPS', 'EPS']
    # Momentum factors, describing moving averages, smooth curves, returns, growth rates, future trend prediction
    momentum_factors = ['REVS10', 'REVS10', 'REVS5', 'RSTR12', 'RSTR24', 'DAREC', 'GREC', 'DAREV', 'GREV', 'DASREV', 'GSREV', 'EARNMOM', 'FiftyTwoWeekHigh', 'BIAS10', 'BIAS20', 'BIAS5', 'BIAS60', 'CCI10''CCI20', 'CCI5', 'CCI88', 'ROC6', 'ROC20', 'SRMI', 'ChandeSD', 'ChandeSU', 'CMO', 'ARC', 'AD', 'AD20', 'AD6', 'CoppockCurve', 'Aroon', 'AroonDown', 'AroonUp', 'DEA', 'DIFF', 'DDI', 'DIZ', 'DIF', 'PVT', 'PCT6', 'PVT12', 'TRIX5', 'TRIX10', 'MA10RegressCoeff12', 'MA10RegressCoeff6', 'PLRC6', 'PLRC12', 'APBMA', 'BBIC', 'MA10Close', 'BearPower', 'RC12', 'RC24']
    # Growth class, calculating growth rates
    growth_factors = ['EGRO', 'FinancingCashGrowRate', 'InvestCashGrowRate', 'NetAssetGrowRate', 'NetProfitGrowRate', 'NPParentCompanyGrowRate', 'OperatingProfitGrowRate', 'OperatingRevenueGrowRate', 'OperCashGrowRate', 'SUE', 'TotalAssetGrowRate', 'TotalProfitGrowRate', 'REC', 'FEARNG', 'FSALESG', 'SUOI']
    
    temp = list()
    temp.append(quality_factors)
    temp.append(return_risk_factors)
    temp.append(valuation_factors)
    temp.append(sentiment_factors)
    temp.append(technical_indicators)
    temp.append(momentum_factors)
    temp.append(growth_factors)

    return temp


def type_filter(score, score_name, threshold=0.1):
    # threshold: represents the ratio of the number of this type
    
    type_list = form_us_sector_types()
    
    type_list_filtered = []
    
    df = pd.DataFrame({'score': score, 'score_name': score_name})
    
    for i in range(len(type_list)):
        now_type = type_list[i]
        df.loc[df.loc[:, 'score_name'].isin(now_type), 'type'] = i+1
            
    df.fillna(0.0, inplace=True)
    
    df = df.sort_values(by=['type', 'score'], ascending=False)
   
    for i in range(len(type_list)):
        now_n = np.int(threshold * len(type_list[i]))+1
        now_type = [i for i in df.loc[df['type'] == i+1, 'score_name'][0:now_n].values]
        type_list_filtered.append(now_type)
        
    return type_list_filtered


def solve1(dataset, lengths, feature_col, feature_list):
    # Based on the sample size of dataset, forms a result with the same sample size and same sorting, but some may not be usable
    # So also returns an allow_flag that records whether the data has been processed
    # input:
    #     dataset
    #     lengths
    #     feature_col: corresponding column names of dataset
    #     feature_list: feature names that need to be processed
    # output:
    #     result: processed data
    #     allow_flag: records whether this data can be used

    result = np.zeros((dataset.shape[0], len(feature_list)))
    allow_flag = np.zeros(len(dataset))

    temp_result = np.zeros((dataset.shape[0], len(feature_list)))
    for i in range(len(feature_list)):
        now_feature = feature_list[i]
        temp_result[:, i] = dataset[:, feature_col.index(now_feature)]

    for i in range(len(lengths)):
        begin_index, end_index = get_us_stock_index(lengths, i)

        now_dataset = temp_result[begin_index:end_index]

        temp = now_dataset[3:] - now_dataset[0:-3]

        temp = np.row_stack((np.zeros((3, now_dataset.shape[1])), temp))

        result[begin_index:end_index] = temp
        allow_flag[begin_index + 3:end_index] = 1

    ss = StandardScaler()
    result = ss.fit_transform(result) * 10
    # Remove some abnormal results
    result[result >= 5] = 5
    result[result <= -5] = -5

    return result, allow_flag


def solve2(dataset, feature_col, feature_list):
    # Based on the sample size of dataset, forms a result with the same sample size and same sorting, but some may not be usable
    # So also returns an allow_flag that records whether the data has been processed
    # input:
    #     dataset
    #     lengths
    #     feature_col: corresponding column names of dataset
    #     feature_list: feature names that need to be processed
    # output:
    #     result: processed data
    #     allow_flag: records whether this data can be used

    result = np.ones((dataset.shape[0], len(feature_list)))

    for i in range(len(feature_list)):
        result[:, i] = dataset[:, feature_col.index(feature_list[i])]

    allow_flag = np.ones(dataset.shape[0])

    return result, allow_flag


def process_us_stock_data(dataset, lengths, feature_col, feature_list):
    # Based on the sample size of dataset, forms a result with the same sample size and same sorting, but some may not be usable
    # So also returns an allow_flag that records whether the data has been processed
    # input:
    #     dataset
    #     lengths
    #     feature_col: corresponding column names of dataset
    #     feature_list: feature names that need to be processed
    # output:
    #     result: processed data
    #     allow_flag: records whether this data can be used

    result1, allow_flag1 = solve1(dataset, lengths, feature_col, feature_list)
    result2, allow_flag2 = solve2(dataset, feature_col, feature_list)

    result = np.column_stack((result1, result2))
    allow_flag = combine_us_stock_data_us_stock_data_us_stock_flags(allow_flag1, allow_flag2)
    
    return result, allow_flag


def form_model(X, lengths, n, v_type, n_iter, verbose=True):
    model = hmm.GaussianHMM(n_components=n, covariance_type=v_type, n_iter=n_iter, verbose=verbose).fit(X, lengths)
    return model


if __name__ == '__main__':
    
    score, feature_name = load_us_stock_single_score()
    list_by_diff_type = type_filter(score, feature_name, 0.1)
    model_list = []
    
    for i in range(len(list_by_diff_type)):
        
        feature_col = list_by_diff_type[i]
        if len(feature_col) == 0:
            continue
    
        dataset, label, lengths, col_nan_record = form_us_stock_dataset(feature_col, label_length=3)
        
        solved_dataset, allow_flag = process_us_stock_data(dataset, lengths, feature_col, feature_col)
        
        X_train, label_train, lengths_train = form_us_stock_model_dataset(solved_dataset, label, allow_flag, lengths)

        X_train = handle_us_stock_outliers(X_train, lengths_train)

        print(X_train.shape)
        model = form_model(X_train, lengths_train, 6, 'diag', 1000)
        model_list.append(model)
        
    pickle.dump(model_list, open('save/HMM_us_stocks_model_list.csv', 'wb'))
