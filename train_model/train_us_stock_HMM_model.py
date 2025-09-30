from dataset_code.process_us_stock_raw_data import form_us_stock_dataset
from public_tool.form_us_stock_model_dataset import form_us_stock_model_dataset
from public_tool.handle_us_stock_outliers import handle_us_stock_outliers
from XGB_HMM.US_stock_GMM_HMM import GMM_HMM
from train_model.US_stock_XGB_HMM import XGB_HMM
import pickle
from dataset_code import HMM_us_stocks, HMM_us_market
import os


def train_HMM_model(n_states):
    # train the hnagqing or the multi_factor GMM_HMM model and XGB_HMM model

    # 1 market
    # 1.1 generate the market dataset
    if not (os.path.exists('data/us_stocks/market_GMM_HMM_model.csv') and os.path.exists('data/us_stocks/market_XGB_HMM_model.csv')):
        feature_col = ['Prev_Close', 'Open', 'Close', 'Volume', 'High', 'Low']
        dataset, label, lengths, col_nan_record = form_us_stock_dataset(feature_col, label_length=5)
        solved_dataset, allow_flag = HMM_us_market.process_us_stock_data(dataset, lengths, feature_col)
        X_train, y_train, lengths_train = form_us_stock_model_dataset(solved_dataset, label, allow_flag, lengths)
        X_train = handle_us_stock_outliers(X_train, lengths_train)

        # 1.2 train and save the GMM_HMM model
        print('training market GMM_HMM model...')
        temp = GMM_HMM(X_train, lengths_train, n_states, 'diag', 1000, True)
        pickle.dump(temp, open('data/us_stocks/market_GMM_HMM_model.csv', 'wb'))

        # 1.3 train and save the XGB_HMM model
        print('training market XGB_HMM model...')
        A, xgb_model, pi = XGB_HMM(X_train, lengths_train)
        pickle.dump([A, xgb_model, pi], open('data/us_stocks/market_XGB_HMM_model.csv', 'wb'))

    # 2 multi_factor
    print('training multi_factor...')
    if not (os.path.exists('data/us_stocks/multi_factor_GMM_HMM_model.csv') and os.path.exists('data/us_stocks/multi_factor_XGB_HMM_model.csv')):
        score, feature_name = HMM_us_stocks.load_us_stock_single_score()
        feature_col_multi_factor = HMM_us_stocks.type_filter(score, feature_name, 0.1)  # there are 7 kinds of multi_factor
        GMM_model_list = []
        XGB_model_list = []
        for i in range(len(feature_col_multi_factor)):
            feature_col = feature_col_multi_factor[i]
            # 2.1 generate the multi_factor dataset
            dataset, label, lengths, col_nan_record = form_us_stock_dataset(feature_col, label_length=5)
            print(sum(lengths))
            print(dataset.shape[0])
            solved_dataset, allow_flag = HMM_us_stocks.process_us_stock_data(dataset, lengths, feature_col, feature_col)
            X_train, label_train, lengths_train = form_us_stock_model_dataset(solved_dataset, label, allow_flag, lengths)
            pickle.dump([X_train, lengths_train], open('data/us_stocks/temp.csv', 'wb'))
            X_train = handle_us_stock_outliers(X_train, lengths_train)

            # 2.2 train and record the GMM_HMM model
            print('training multi_factor GMM_HMM model %s...' % (i+1))
            pickle.dump([X_train, lengths_train], open('data/us_stocks/temp1.csv', 'wb'))
            print(X_train.shape[0])
            print(sum(lengths_train))
            temp = GMM_HMM(X_train, lengths_train, n_states, 'diag', 1000, True)
            GMM_model_list.append(temp)

            # 2.3 train and record the XGB_HMM model
            print('training multi_factor XGB_HMM model %s...' % (i+1))
            A, xgb_model, pi = XGB_HMM(X_train, lengths_train)
            XGB_model_list.append([A, xgb_model, pi])

        # 2.4 save the model
        pickle.dump(GMM_model_list, open('data/us_stocks/multi_factor_GMM_HMM_model.csv', 'wb'))
        pickle.dump(XGB_model_list, open('data/us_stocks/multi_factor_XGB_HMM_model.csv', 'wb'))
