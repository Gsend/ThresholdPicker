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
   
2. In case you need to optimize the return on investment(ROI) 
of your model. with a given value for each True-Positive and 
a cost for each False-Positive. 
This module can be run to return the threshold returns the 
maximal ROI on your validation set. 
Thus allowing you to fine tune your model for maximal ROI.
   
