import numpy as np


def calculate_us_stock_accuracy(X, y, model):
    # Calculateof，yas-1,0,1of
    # Returnsof，and

    single_record = np.zeros(3)

    pred_proba = model.predict(X)

    for i in range(pred_proba.shape[0]):
        temp = pred_proba[i, :]
        pred_class = np.argmax(temp)
        true_class = np.where(y[i, :] == 1)[0][0]

        if pred_class == true_class:
            single_record[true_class] += 1

    acc = sum(single_record) / len(y)

    for i in range(len(single_record)):
        single_record[i] = single_record[i] / sum(y[:, i] == 1)

    return acc, single_record
