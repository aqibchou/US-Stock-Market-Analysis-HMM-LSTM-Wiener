import numpy as np
from public_tool.combine_us_stock_data_us_stock_data_us_stock_flags import combine_us_stock_data_us_stock_data_us_stock_flags
from public_tool.get_us_stock_index import get_us_stock_index
import warnings
warnings.filterwarnings("ignore")


def solve1(dataset, lengths, feature_col):
    # Based ondatasetofsample，Formsample，ofProcessresult，isisof
    # ReturnsaRecorddataisProcessofallow_flag
    # input:
    #     dataset, feature_col
    #     lengths
    # output:
    #     result: Processofdata
    #     allow_flag: Recordthatdatais

    result = np.zeros((dataset.shape[0], 3))
    allow_flag = np.zeros(dataset.shape[0])

    for i in range(len(lengths)):
        begin_index, end_index = get_us_stock_index(lengths, i)

        Close = dataset[begin_index:end_index, feature_col.index('Close')]
        vol = dataset[begin_index:end_index, feature_col.index('Volume')]
        High = dataset[begin_index:end_index, feature_col.index('High')]
        Low = dataset[begin_index:end_index, feature_col.index('Low')]

        logDel = np.log(High) - np.log(Low)
        logRet_5 = np.log(Close[5:]) - np.log(Close[:-5])
        logVol_5 = np.log(vol[5:]) - np.log(vol[:-5])

        logRet_5 = np.hstack((np.zeros(5), logRet_5))
        logVol_5 = np.hstack((np.zeros(5), logVol_5))

        temp = np.column_stack((logDel, logRet_5, logVol_5))
        result[begin_index:end_index, :] = temp

        temp = np.hstack((np.zeros(5), np.ones(end_index - begin_index - 5)))
        allow_flag[begin_index:end_index] = temp

    return result, allow_flag


def solve2(dataset, feature_col):
    # Based ondatasetofsample，Formsample，ofProcessresult，isisof
    # ReturnsaRecorddataisProcessofallow_flag
    # input:
    #     dataset, feature_col
    #     lengths
    # output:
    #     result: Processofdata
    #     allow_flag: Recordthatdatais

    result = np.zeros((dataset.shape[0], 4))
    result[:, 0] = dataset[:, feature_col.index('Open')]/dataset[:, feature_col.index('Prev_Close')]
    result[:, 1] = dataset[:, feature_col.index('Low')]/dataset[:, feature_col.index('Prev_Close')]
    result[:, 2] = dataset[:, feature_col.index('High')]/dataset[:, feature_col.index('Prev_Close')]
    result[:, 3] = dataset[:, feature_col.index('Close')]/dataset[:, feature_col.index('Prev_Close')]

    allow_flag = np.ones(result.shape[0])

    return result, allow_flag


def process_us_stock_data(dataset, lengths, feature_col):
    # Based ondatasetofsample，Formsample，ofProcessresult，isisof
    # ReturnsaRecorddataisProcessofallow_flag
    # input:
    #     dataset, feature_col
    #     lengths
    # output:
    #     result: Processofdata
    #     allow_flag: Recordthatdatais

    result1, allow_flag1 = solve1(dataset, lengths, feature_col)
    result2, allow_flag2 = solve2(dataset, feature_col)

    result = np.column_stack((result1, result2))
    allow_flag = combine_us_stock_data_us_stock_data_us_stock_flags(allow_flag1, allow_flag2)

    return result, allow_flag
