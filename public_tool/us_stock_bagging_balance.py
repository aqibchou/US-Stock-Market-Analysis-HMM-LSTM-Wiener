import numpy as np
import random


def us_stock_bagging_balance(X, y):
# classdataclassofdata，baggingofdata
# input:
#         X, y
#         InputofXisor，issample_num
#         yis，isone_hotof
# output:
#         X_result, y_result
#         OutputofandInputofX, y

    drop_th = 0.01  # classratiothreshold，asthatclass
    max_subsample_ratio = 1  # Recordindataofmax_n_subsample，baggingofdataofclassofcountismax_n_subsample*thatParameters

    if y.ndim == 1:
        y_label = y
    else:
        y_label = np.zeros(y.shape[0])
        for i in range(y.shape[0]):
            y_label[i] = np.where(y[i] == 1)[0][0]

    unique = np.unique(y_label)
    num_class = len(unique)
    unique_ratio = np.zeros(num_class)
    for i in range(num_class):
        unique_ratio[i] = sum(y_label == unique[i]) / len(y_label)

    unique_ratio[unique_ratio < drop_th] = 0

    n_bagging = int(max(unique_ratio) * len(y) * max_subsample_ratio)

    X_result = []
    y_result = []
    for i in range(num_class):
        if unique_ratio[i] == 0:
            continue
        else:
            sub_X = X[y_label == unique[i]]
            sub_y = y[y_label == unique[i]]
            for j in range(n_bagging):
                index = random.randint(0, sub_X.shape[0] - 1)
                X_result.append(sub_X[index])
                y_result.append(sub_y[index])
    X_result = np.array(X_result)
    y_result = np.array(y_result)
    temp = [i for i in range(X_result.shape[0])]
    random.shuffle(temp)
    X_result = X_result[temp]
    y_result = y_result[temp]

    return X_result, y_result
