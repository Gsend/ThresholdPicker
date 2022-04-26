import numpy as np


def get_hist_cumsum(proba_array, num_bins):
    hist_, bins = np.histogram(proba_array, bins=num_bins)
    return np.cumsum(hist_), bins[1:]



def get_recall_threshold_curve(pos_proba_array, num_bins=100):
    num_true = len(pos_proba_array)
    hist_cumsum, bins = get_hist_cumsum(pos_proba_array, num_bins)
    true_pos = num_true - hist_cumsum
    false_neg = hist_cumsum
    return true_pos/num_true, bins


def get_precission_threshold_curve(pos_proba_array, neg_proba_array, num_bins=100):


if __name__ == '__main__':
    pass