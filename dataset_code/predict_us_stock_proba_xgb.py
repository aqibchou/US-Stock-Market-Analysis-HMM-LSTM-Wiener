import numpy as np
from public_tool.get_us_stock_index import get_us_stock_index
from XGB_HMM.form_us_stock_B_matrix_by_XGB import form_B_matrix_by_XGB
from XGB_HMM.predict_us_stock import self_pred


def predict_us_stock_proba_xgb(A, model, pi, O, allow_flag, lengths):
    # datasetFormpred_proba，ofdatasetisprocess_us_stock_dataof，allow_flagofdata
    # output:
    #     pred_proba：countclass

    n_states = len(pi)
    pred_proba = np.zeros((O.shape[0], n_states))

    for i in range(len(lengths)):
        begin_index, end_index = get_us_stock_index(lengths, i)

        now_O = O[begin_index:end_index, :]
        now_allow_flag = allow_flag[begin_index:end_index]
        now_pred_proba = np.zeros((now_O.shape[0], n_states))

        now_allow_B = form_B_matrix_by_XGB(model, now_O[now_allow_flag == 1], pi)
        _, now_allow_pred_proba, _ = self_pred(now_allow_B, [now_allow_B.shape[0]], A, pi)

        now_pred_proba[now_allow_flag == 1] = now_allow_pred_proba
        pred_proba[begin_index:end_index] = now_pred_proba

    return pred_proba
