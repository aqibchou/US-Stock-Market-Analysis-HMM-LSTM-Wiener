"""
    ofdata，astock idaofdata
"""

import os
import pandas as pd
import numpy as np 
import pickle

# data
file_path = 'C:/Users/Administrator/Desktop/program/data/market/'
file_list = os.listdir(file_path)
columns_name = pd.read_csv(file_path+file_list[0]).columns

market_record = []
temp_record = pd.DataFrame(columns=columns_name)
for i in range(len(file_list)):
    now_path = file_path+file_list[i]
    now_df = pd.read_table(now_path, sep=',')
    temp_record = pd.concat((temp_record, now_df), axis=0)
    if (i+1) % 50 == 0 or (i+1) == len(file_list):
        del temp_record['Unnamed: 0']
        del temp_record['Unnamed: 25']
        market_record.append(temp_record)
        temp_record = pd.DataFrame(columns=columns_name)
    print('all:%s, now:%s' % (len(file_list), i+1))

for i in range(len(market_record)):
    if i == 0:
        market_df = market_record[0]
    else:
        market_df = pd.concat((market_df, market_record[i]), axis=0)
del market_record

# factor
file_path = 'C:/Users/Administrator/Desktop/program/data/multi_factor/'
file_list = os.listdir(file_path)
columns_name = pd.read_csv(file_path+file_list[0]).columns

multi_factor_record = []
temp_record = pd.DataFrame(columns=columns_name)
for i in range(len(file_list)):
    now_file = file_list[i]
    now_path = file_path+now_file
    tradeDate = now_file[0:4]+'-'+now_file[4:6]+'-'+now_file[6:8]
    now_df = pd.read_table(now_path, sep=',')
    now_df['tradeDate'] = tradeDate
    temp_record = pd.concat((temp_record, now_df), axis=0)
    if (i+1) % 30 == 0 or (i+1) == len(file_list):
        del temp_record['Unnamed: 0']
        del temp_record['Unnamed: 248']
        multi_factor_record.append(temp_record)
        temp_record = pd.DataFrame(columns=columns_name)
    print('all:%s, now:%s' % (len(file_list), i+1))


# ofFormIDasaofdata
unique_id = np.unique(market_df['secID'].values)

multi_factor_columns = multi_factor_record[0].columns
for i in range(len(unique_id)):
    now_id = unique_id[i]
    now_market_df = market_df[market_df['secID'] == now_id]
    now_multi_factor_df = pd.DataFrame(columns=multi_factor_columns)
    for temp in multi_factor_record:
        now_temp = temp[temp['secID'] == now_id]
        if now_temp.shape[0] != 0:
            now_multi_factor_df = pd.concat((now_multi_factor_df, now_temp), axis=0)
    now_df = pd.merge(now_market_df, now_multi_factor_df, on=['secID', 'tradeDate'], how='left')
    pickle.dump(now_df, open('save/us_stocks_by_sector/'+now_id+'.csv', 'wb'))
    print('all:%s, now:%s' % (len(unique_id), i+1))
