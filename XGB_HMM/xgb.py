import xgboost as xgb
import numpy as np


def self_xgb(X, gamma, n_states):

    params = {'objective': 'multi:softprob',
              'learning_rate': 0.01,
              'colsample_bytree': 0.886,
              'min_child_weight': 3,
              'max_depth': 10,
              'subsample': 0.886,
              'reg_alpha': 1.5,  # L1 regularization coefficient
              'reg_lambda': 0.5,  # L2 regularization coefficient
              'gamma': 0.5,  # Minimum loss reduction for split
              'n_jobs': -1,
              'eval_metric': 'mlogloss',
              'scale_pos_weight': 1,
              'random_state': 201806,
              'missing': None,
              'silent': 1,
              'max_delta_step': 0,
              'num_class': n_states}

    y = np.array([np.argmax(i) for i in gamma])
    temp = np.array([np.max(i) for i in gamma])
    
    # Use a lower threshold if no samples meet 0.9
    threshold = 0.9
    if np.sum(temp >= threshold) == 0:
        threshold = 0.5
        print(f"No samples met 0.9 threshold, using {threshold} instead")
    
    valid_mask = temp >= threshold
    if np.sum(valid_mask) == 0:
        # If still no valid samples, use all samples
        print("No samples met threshold, using all samples")
        valid_mask = np.ones(len(y), dtype=bool)
    
    y = y[valid_mask]
    X = X[valid_mask]
    sample_weight = temp[valid_mask]

    d_train = xgb.DMatrix(X, y, weight=sample_weight)

    model = xgb.train(params, d_train, num_boost_round=1000, verbose_eval=True)

    pred = np.array([np.argmax(i) for i in model.predict(d_train)])
    if len(y) > 0:
        print(sum(pred == y)/len(y))
    else:
        print("No valid samples for accuracy calculation")

    return model
