import numpy as np
from public_tool.get_us_stock_index import get_us_stock_index


def predict_us_stock_proba_gmm(model, O, allow_flag, lengths):
    # datasetFormpred_proba，ofdatasetisprocess_us_stock_dataof，allow_flagofdata
    # output:
    #     pred_proba：countclass

    pred_proba = np.zeros((O.shape[0], model.n_components))

    for i in range(len(lengths)):
        begin_index, end_index = get_us_stock_index(lengths, i)

        now_O = O[begin_index:end_index, :]
        now_allow_flag = allow_flag[begin_index:end_index]
        now_pred_proba = np.zeros((now_O.shape[0], model.n_components))

        now_pred_proba[now_allow_flag == 1] = model.predict_proba(now_O[now_allow_flag == 1])

        pred_proba[begin_index:end_index] = now_pred_proba

    return pred_proba
