import sys
import numpy as np


def combine_us_stock_data_us_stock_data_us_stock_flags(allow_flag1, allow_flag2):
    # allow_flag，as1as1

    if not len(allow_flag1) == len(allow_flag2):
        sys.exit('length of two allow_flag is not equal')
    result = np.zeros(len(allow_flag1))
    for i in range(len(allow_flag1)):
        if allow_flag1[i] == 1 and allow_flag2[i] == 1:
            result[i] = 1
        else:
            result[i] = 0
    return result
