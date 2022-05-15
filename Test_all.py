import unittest
from utils import *
np.random.seed(0)

class TestMethods(unittest.TestCase):

    def test_recall_curve(self):
        positive_probas = [.5, .6, .9, 1, 0.1]
        recall, thresholds = get_recall_threshold_curve(positive_probas, num_bins=4)
        self.assertListEqual(list(recall), [.8, .6, .4, 0])

    def test_precission_curve(self):
        num_bins = 100
        predicted_probas = np.arange(0, 1 ,.01)
        labels = np.random.choice([0,1], num_bins)
        percission, thresholds = get_precision_threshold_curve(predicted_probas, labels, num_bins=100)
        self.assertAlmostEqual(percission.mean(), .5, delta=.05)
        self.assertEquals(len(num_bins), len(thresholds))

if __name__ == '__main__':
    unittest.main()