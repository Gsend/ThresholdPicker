from ThresholdPicker.utils import get_recall_threshold_curve, \
    get_precision_threshold_curve, get_all_rates
import numpy as np


class ThresholdPicker():

    def __init__(self, num_bins=1000, beta=2):
        self.num_bins = num_bins
        self.beta = beta
        self.values = None


    def get_curves(self, probas, labels):
        recall_rate, _ = get_recall_threshold_curve(probas[labels > 0],
                                                    num_bins=self.num_bins)
        precision_rate, bins = get_precision_threshold_curve(probas, labels,
                                                             num_bins=self.num_bins)
        return recall_rate, precision_rate, bins

    def get_threshold(self, probas, labels, target, mode='recall', beta=1):
        """
        calculate the threshold that would return the closest valuse to
        the desired recall/percision.
        :param target: float, between 0 and 1  the desired recall/percision/fscore
        :param mode: str, defines wheather to return threhsold for recall, percision or
        :param betta: float, the betta value for fscore calculation default is 2.
        fscore
        :return: threshold float, between 0 and 1, score float between 0 and 1.
        """
        if mode == 'recall':
            recall_sig, thresholds = get_recall_threshold_curve(probas[labels > 0])
            diffs = np.abs(recall_sig - target)
            min_index = np.argmin(diffs)
            return thresholds[min_index], recall_sig[min_index]

        if mode == 'percision':
            percision_sig, thresholds = get_recall_threshold_curve(
                probas[labels > 0], num_bins=self.num_bins)
            diffs = np.abs(percision_sig - target)
            min_index = np.argmin(diffs)
            return thresholds[min_index], percision_sig[min_index]

        if mode == 'fscore':
            recall_sig, percision_sig, thresholds = self.get_curves(probas, labels)
            fscores = self.calc_fscore(percision_sig, recall_sig, beta)
            diffs = np.abs(fscores - target)
            min_index = np.argmin(diffs)
            return thresholds[min_index], fscores[min_index], fscores


    @staticmethod
    def calc_fscore(percision_sig, recall_sig, beta):
        return (1 + beta**2) * (percision_sig * recall_sig) \
               / (beta ** 2 * percision_sig + recall_sig)

    def gen_value_signal(self, probas, labels,
                               tp_value, fp_value, tn_value, fn_value):
        all_signals = get_all_rates(probas, labels, num_bins=self.num_bins)
        tp = all_signals['TP']
        fp = all_signals['FP']
        tn = all_signals['TN']
        fn = all_signals['FN']
        thresholds = all_signals['thresholds']
        return tp*tp_value + fp*fp_value + tn*tn_value + fn*fn_value, thresholds

    def gen_optimal_return_threshold(self, probas, labels, tp_value,
                                     fp_value, tn_value, fn_value):
        true_rate = labels.mean()
        recall_sig, percision_sig, thresholds = self.get_curves(probas, labels)
        return_signal, thresholds = self.gen_value_signal(probas, labels, tp_value,
                                                       fp_value, tn_value, fn_value)

        best_threshold_index = np.argmax(return_signal)
        return thresholds[best_threshold_index], return_signal