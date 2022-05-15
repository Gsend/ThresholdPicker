from utils import get_recall_threshold_curve, get_precision_threshold_curve
import numpy as np


class PrecisionRecallThresholdCurve():

    def __init__(self, num_bins=1000, betta=2):
        self.num_bins = num_bins
        self.betta = betta
        self.values = None


    def get_curves(self, probas, labels):
        recall_rate, _ = get_recall_threshold_curve(probas[labels > 0],
                                                    num_bins=self.num_bins)
        precision_rate, bins = get_precision_threshold_curve(probas, labels,
                                                             num_bins=self.num_bins)
        return recall_rate, precision_rate, bins

    def get_threshold(self, probas, labels, target, mode='recall', betta=1):
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
            recalls, thresholds = get_recall_threshold_curve(probas[labels > 0])
            diffs = np.abs(recalls - target)
            min_index = np.argmin(diffs)
            return thresholds[min_index], recalls[min_index]

        if mode == 'percision':
            percisions, thresholds = get_recall_threshold_curve(
                probas[labels > 0], num_bins=self.num_bins)
            diffs = np.abs(percisions - target)
            min_index = np.argmin(diffs)
            return thresholds[min_index], percisions[min_index]

        if mode == 'fscore':
            percisions, recalls, thresholds = self.get_curves(probas, labels)
            fscores = self.calc_fscore(percisions, recalls, betta)
            min_index = np.argmin(fscores - target)
            return thresholds[min_index], fscores[min_index]

    @staticmethod
    def calc_fscore(percisions, recalls, betta):
        return (1 + betta**2) * (percisions * recalls) / (betta ** 2 * percisions + recalls)
