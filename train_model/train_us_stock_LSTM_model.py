from dataset_code.process_us_stock_raw_data import form_us_stock_dataset
import pickle
from dataset_code import HMM_us_stocks, HMM_us_market
from dataset_code.predict_us_stock_proba_gmm import predict_us_stock_proba_gmm
from dataset_code.combine_us_stock_data_us_stock_data_us_stock_data import combine_us_stock_data_us_stock_data
from dataset_code.predict_us_stock_proba_xgb import predict_us_stock_proba_xgb
from train_model.US_stock_LSTM import self_LSTM


def train_LSTM_model():
    # train the LSTM model based on state_proba formed by GMM_HMM or XGB_HMM

    # generate the dataset

    feature_col_market = ['Prev_Close', 'Open', 'Close', 'Volume', 'High', 'Low']
    score, feature_name = HMM_us_stocks.load_us_stock_single_score()
    feature_col_multi_factor = HMM_us_stocks.type_filter(score, feature_name, 0.1)
    feature_col = feature_col_market
    _ = [[feature_col.append(j) for j in i] for i in feature_col_multi_factor]
    dataset, label, lengths, col_nan_record = form_us_stock_dataset(feature_col, 5)

    # 1 by GMM_HMM
    # 1.1 market
    solved_dataset1, allow_flag1 = HMM_us_market.process_us_stock_data(dataset, lengths, feature_col)
    model = pickle.load(open('data/us_stocks/market_GMM_HMM_model.csv', 'rb'))
    pred_proba1 = predict_us_stock_proba_gmm(model, solved_dataset1, allow_flag1, lengths)

    # 1.2 multi_factor
    pred_proba2 = []
    allow_flag2 = []
    model = pickle.load(open('data/us_stocks/multi_factor_GMM_HMM_model.csv', 'rb'))
    for i in range(len(feature_col_multi_factor)):
        temp_solved_dataset, temp_allow_flag = HMM_us_stocks.process_us_stock_data(dataset, lengths, feature_col, feature_col_multi_factor[i])
        temp_model = model[i]
        temp_pred_proba = predict_us_stock_proba_gmm(temp_model, temp_solved_dataset, temp_allow_flag, lengths)
        pred_proba2.append(temp_pred_proba)
        allow_flag2.append(temp_allow_flag)

    # 1.3 combine_us_stock_data_us_stock_data two type state_proba
    final_X, final_y, final_lengths = combine_us_stock_data_us_stock_data(pred_proba1, pred_proba2, allow_flag1, allow_flag2, label, lengths)

    # 1.4 train LSTM model
    self_LSTM(final_X, final_y, final_lengths, 'GMM_HMM_LSTM')

    # 2 by XGB_HMM
    # 2.1 market
    solved_dataset1, allow_flag1 = HMM_us_market.process_us_stock_data(dataset, lengths, feature_col)
    temp = pickle.load(open('data/us_stocks/market_XGB_HMM_model.csv', 'rb'))
    A, model, pi = temp[0], temp[1], temp[2]
    pred_proba1 = predict_us_stock_proba_xgb(A, model, pi, solved_dataset1, allow_flag1, lengths)

    # 2.2 multi_factor
    pred_proba2 = []
    allow_flag2 = []
    model = pickle.load(open('data/us_stocks/multi_factor_XGB_HMM_model.csv', 'rb'))
    for i in range(len(feature_col_multi_factor)):
        temp_solved_dataset, temp_allow_flag = HMM_us_stocks.process_us_stock_data(dataset, lengths, feature_col, feature_col_multi_factor[i])
        temp_A, temp_model, temp_pi = model[i][0], model[i][1], model[i][2]
        temp_pred_proba = predict_us_stock_proba_xgb(temp_A, temp_model, temp_pi, temp_solved_dataset, temp_allow_flag, lengths)
        pred_proba2.append(temp_pred_proba)
        allow_flag2.append(temp_allow_flag)

    # 2.3 combine_us_stock_data_us_stock_data two type state_proba
    final_X, final_y, final_lengths = combine_us_stock_data_us_stock_data(pred_proba1, pred_proba2, allow_flag1, allow_flag2, label, lengths)

    # 2.4 train LSTM model
    self_LSTM(final_X, final_y, final_lengths, 'XGB_HMM_LSTM')
