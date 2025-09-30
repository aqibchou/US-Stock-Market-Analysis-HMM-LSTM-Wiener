import sys
import numpy as np
from public_tool.get_us_stock_index import get_us_stock_index


def combine_us_stock_data_us_stock_data(X1, X2, allow_flag1, allow_flag2, label, lengths):
# ofdatainofdata
# X
# allow_flagis0ofsample，labelis-2ofsample
# input:
#     X: ofdata，as1array，asarrayofalist
#     allow_flag: isof，as1array，asarrayofalist，andofX
#     label: label
#     lengths: lengths
# output:
#     result_X: ofX，arrayclass
#     result_label: oflabel
#     result_lengths：oflengths

    if not (type(X1) == type(allow_flag1) or type(X2) == type(allow_flag2)):
        sys.exit('x and allow_flagofInput')

    list_flag1 = type(X1) == list
    list_flag2 = type(X2) == list

    X = np.zeros((len(label), 0))
    allow_flag = np.zeros(len(label))
    count = 0

    if list_flag1 == 1:
        for i in range(len(X1)):
            X = np.column_stack(X, X1[i])
            allow_flag += allow_flag1[i]
            count += 1
    else:
        X = np.column_stack((X, X1))
        allow_flag += allow_flag1
        count += 1
    if list_flag2 == 1:
        for i in range(len(X2)):
            X = np.column_stack((X, X2[i]))
            allow_flag += allow_flag2[i]
            count += 1
    else:
        X = np.column_stack((X, X2))
        allow_flag += allow_flag2
        count += 1
    allow_flag[allow_flag < count] = 0
    allow_flag[allow_flag == count] = 1

    result_X = np.zeros((0, X.shape[1]))
    result_label = np.zeros(0)
    result_lengths = []

    for i in range(len(lengths)):
        begin_index, end_index = get_us_stock_index(lengths, i)

        now_X = X[begin_index:end_index]
        now_allow_flag = allow_flag[begin_index:end_index]
        now_label = label[begin_index:end_index]

        temp = np.logical_and(now_allow_flag == 1, now_label != -2)

        result_X = np.row_stack((result_X, now_X[temp]))
        result_label = np.hstack((result_label, now_label[temp]))
        result_lengths.append(sum(temp))

    return result_X, result_label, result_lengths
