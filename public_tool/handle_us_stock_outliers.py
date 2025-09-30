from public_tool.get_us_stock_index import get_us_stock_index
import numpy as np
from dataset_code.process_us_stock_raw_data import fill_na


def handle_us_stock_outliers(dataset, lengths):
    """
    Find the outlier data, and replace them by fill_na function
    input:
        dataset, array
        lengths, list, Record the length of chains
    output:
        dataset, array
    """

    n = 3     # ifisthanmeandifferentnunitofstandard deviation，thenjudgeasoutlier
    result = np.zeros(dataset.shape)
    for i in range(len(lengths)):
        begin_index, end_index = get_us_stock_index(lengths, i)
        for j in range(dataset.shape[1]):
            temp = dataset[begin_index:end_index, j].copy()
            if max(temp) > 4.5:
                flag = 1
            mean = np.mean(temp)
            std = np.std(temp)
            temp[np.logical_or(temp >= mean+n*std, temp <= mean-n*std)] = np.mean(temp)
# temp = fill_na(temp, 100)
            result[begin_index:end_index, j] = temp

    return result
