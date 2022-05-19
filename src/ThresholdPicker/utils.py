import numpy as np
import pandas as pd

def get_hist_cumsum(proba_array, num_bins):
    hist_, bins = np.histogram(proba_array, bins=num_bins)
    return np.cumsum(hist_)[:-1], bins[0:-1]


def get_recall_threshold_curve(pos_proba_array, num_bins=100):
    num_true = len(pos_proba_array)
    hist_cumsum, bins = get_hist_cumsum(pos_proba_array, num_bins)
    true_pos = num_true - hist_cumsum
    return true_pos/num_true, bins


def get_all_rates(predicted_probas, labels, num_bins=100):
    probas_and_labels = pd.DataFrame({'probas': predicted_probas, 'labels': labels})
    true_probas = probas_and_labels['probas'][probas_and_labels['labels'] == 1]
    num_true = len(true_probas)
    false_probas = probas_and_labels['probas'][probas_and_labels['labels'] == 0]
    num_false = len(false_probas)
    false_neg, _ = get_hist_cumsum(true_probas, num_bins)
    true_neg, bins = get_hist_cumsum(false_probas, num_bins)
    true_pos = num_true - false_neg
    false_pos = num_false - true_neg
    return {'TP': true_pos, 'FP': false_pos,
            'TN': true_neg, 'FN': false_neg, 'thresholds': bins}


def get_precision_threshold_curve(predicted_probas, labels, num_bins=100, epsilon=1e-10):
    rates = get_all_rates(predicted_probas, labels, num_bins)
    true_pos = rates['TP']
    false_pos = rates['FP']
    precission = (true_pos + epsilon)/(true_pos + false_pos + epsilon)
    return precission, rates['threshold_bins']


if __name__ == '__main__':
    pass