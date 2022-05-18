# ThresholdPicker
## a Tool for optimize Threshold moving for Binary Classifiers.

The Purpose of this project is to provide an easy tool for picking the 
best Threshold for a given Binary Classifier.

The tool currently supports the following scenarios:
1. pick threshold by Recall, precision or f-score:
   in case you need to configure the model to return a 
   specific score (recall, Precision or f-score). 
   You can run the ThresholdPicker with a labeled Validation 
   set and receive the threshold that would give the score 
   closest to the the one you specified.
   
   
2. In case you need to balance the True-Positive and
False-Positive rates for optimal performance for a given 
data distribution. For example if the value of each TP is 10$ and the 
cost of every FP is 2$ and your data has 20% True vs 80% False labels.
You can use this tool to optimize the threshold so that your model will
return the maximal average income. 

    from ThresholdPicker.utils import *
    from ThresholdPicker.ThresholdPicker import ThresholdPicker as PRTC
    prtc = PRTC()
    threshold, _ = prtc.gen_optimal_return_threshold(predicted_probas,
                                                     labels,
                                                     true_pos_value=10,
                                                     false_pos_cost=2
                                                     ) 

   
