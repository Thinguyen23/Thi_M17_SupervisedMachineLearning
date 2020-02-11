# Supervised Machine Learning - Credit Risk Study
These are the following models trained and tested on the [Loan Status Database](https://github.com/Thinguyen23/Thi_M17_SupervisedMachineLearning/blob/master/m17_challenge/LoanStats_2019Q1.csv)
## Random Oversampling Model Performance
### Summary and Analysis 

1. The precision for high_risk credits can be determined as 70/(70+6711)=0.01. A low precision indicates a large number of False Positives - of the 6781 high_risk predictions, 6711 were actually low_risk credits, only 70 were actually high risk. Precision for low_risk credits can be determined as 10393/(10393+31)=1.00. A high precsion indicates a large number of True Positives - of the 10424 low_risk predictions, 10393 were actually low_risk credits.

![randomoversampling.png](https://github.com/Thinguyen23/Thi_M17_SupervisedMachineLearning/blob/master/images/randomoversampling.png)

2. Recall score for high_risk credits is 70/(70+31)=0.69, and for low_risk credits is 10393/(10393+6711)=0.61. A high recall is idicative of a small number of False Negatives as compared to True Positives - of the 101 actual high_risk credits, 31 were predicted as low_risk credits.

3. Balanced acurracy score is the average accuracy score per class. The balanced accuracy score is calculated as (70/101 + 10393/17104)/2=0.65, which means of all 101 actual high_risk credits, 70 were correctly predicted and of all  17104 actual low_risk credits, 10393 were correctly predicted.

### Recommendation
The model may not be the best one to assess credit risk because the balanced accuracy, 0.65, is moderate and the precision for high_risk credits, 0.01, is not good enough to state that the model will be good at classifying credit risks.

## SMOTE Oversampling Model Performance
### Summary and Analysis 

1. The precision for high_risk credits can be determined as 64/(64+5291)=0.01. A low precision indicates a large number of False Positives - of the 5355 high_risk predictions, 5291 were actually low_risk credits, only 64 were actually high_risk. Precision for low_risk credits can be determined as 11813/(11813+37)=1.00 - of the 11850 low_risk predictions, only 37  were actually high_risk credits.

![SMOTE.png](https://github.com/Thinguyen23/Thi_M17_SupervisedMachineLearning/blob/master/images/SMOTE.png)

2. Recall score for high_risk credits is 64/(64+37)=0.63, and for low_risk credits is 11813/(11813+5291)=0.69. A high recall is idicative of a small number of False Negatives as compared to True Positives - of the 101 actual high_risk credits, 37 were predicted as low_risk credits.

3. Balanced acurracy score is the average accuracy score per class. The balanced accuracy score is calculated as (64/101 + 11813/17104)/2=0.66, which means of all 101 actual high_risk credits, 64 were correctly predicted and of all  17104 actual low_risk credits, 11813 were correctly predicted.

### Recommendation
The model may not be the best one to assess credit risk because the balanced accuracy, 0.66, is moderate and the precision for high_risk credits, 0.01, is not good enough to state that the model will be good at classifying credit risks.

## Cluster Centroid Undersampling Model Performance
### Summary and Analysis 

1. The precision for high_risk credits can be determined as 67/(67+10217)=0.01. A low precision indicates a large number of False Positives - of the 10284 high_risk predictions, 10217 were actually low_risk credits, only 67 were actually high risk. Precision for low_risk credits can be determined as 6887/(6887+34)=1.00. A high precsion indicates a large number of True Positives - of the 6921 low_risk predictions, 6887 were actually low_risk credits.

![ClusterCentroids.png](https://github.com/Thinguyen23/Thi_M17_SupervisedMachineLearning/blob/master/images/ClusterCentroids.png)

2. Recall score for high_risk credits is 67/(67+34)=0.66, and for low_risk credits is 6887/(6887+10217)=0.40. A low recall is idicative of a large number of False Negatives as compared to True Positives - of the 17104 actual low_risk credits, 10217 were predicted as high_risk credits.

3. Balanced acurracy score is the average accuracy score per class. The balanced accuracy score is calculated as (67/101 + 6887/17104)/2=0.533, which means of all 101 actual high_risk credits, 67 were correctly predicted and of all  17104 actual low_risk credits, 6887 were correctly predicted.

### Recommendation
The model is not be the best one to assess credit risk because the balanced accuracy, 0.533, is low. The low precision for high_risk credits (0.01) and the low recall for low_risk credits (0.4) are not good enough to state that the model will be good at classifying credit risks.

## SMOTEENN Model Performance
### Summary and Analysis 

1. The precision for high_risk credits can be determined as 73/(73+7412)=0.01. A low precision indicates a large number of False Positives - of the 7485 high_risk predictions, 7412 were actually low_risk credits, only 73 were actually high risk. Precision for low_risk credits can be determined as 9692/(9692+28)=1.00. A high precsion indicates a large number of True Positives - of the 9720 low_risk predictions, 9692 were actually low_risk credits.

![SMOTEENN.png](https://github.com/Thinguyen23/Thi_M17_SupervisedMachineLearning/blob/master/images/SMOTEENN.png)

2. Recall score for high_risk credits is 73/(73+28)=0.72, and for low_risk credits is 9692/(9692+7412)=0.57. A low recall is idicative of a large number of False Negatives as compared to True Positives - of the 17104 actual low_risk credits, 7412 were predicted as high_risk credits.

3. Balanced acurracy score is the average accuracy score per class. The balanced accuracy score is calculated as (73/101 + 9692/17104)/2=0.645, which means of all 101 actual high_risk credits, 73 were correctly predicted and of all  17104 actual low_risk credits, 9692 were correctly predicted.

### Recommendation
The model is not be the best one to assess credit risk because the balanced accuracy, 0.645, is low. The low precision for high_risk credits (0.01) and the low recall for low_risk credits (0.57) are not good enough to state that the model will be good at classifying credit risks.
In summary, I do not recommend any of the above models for reasons listed above.

# Extension

## Balanced Random Forest Model Performance
### Summary and Analysis 

1. The precision for high_risk credits can be determined as 68/(68+1749)=0.04. Number of False Positive, 1749, is still high. However, this shows an improvement from previous models (oversampling, undersampling and combination sampling). Precision for low_risk credits can be determined as 15355/(15355+33)=1.00 - of the 15399 low_risk predictions, 15355 were actually low_risk credits.

![BalancedRandomForest.png](https://github.com/Thinguyen23/Thi_M17_SupervisedMachineLearning/blob/master/images/BalancedRandomForest.png)

2. Recall score for high_risk credits is 68/(68+33)=0.67, and for low_risk credits is 15355/(15355+1749)=0.90. A high recall is idicative of a small number of False Negatives as compared to True Positives - of the 17104 actual low_risk credits, only 1749 were predicted as high_risk credits.

3. Balanced acurracy score is the average accuracy score per class. The balanced accuracy score is calculated as (68/101 + 15355/17104)/2=0.786, which means of all 101 actual high_risk credits, 68 were correctly predicted and of all  17104 actual low_risk credits, 15355 were correctly predicted.

### Recommendation
The model is not the best, but still good to assess credit risk because the balanced accuracy, 0.786, is relatively high. However, the low precision for high_risk credits (0.04) is not good enough to state that the model will be best at classifying credit risks.

## Easy Ensemble AdaBoost Model Performance
### Summary and Analysis 

1. The precision for high_risk credits can be determined as 93/(93+983)=0.09. Number of False Positive, 983, is still high. However, this shows an huge improvement from previous models (BalancedRandomForestClassifier). Precision for low_risk credits can be determined as 16121/(16121+8)=1.00 - of the 16129 low_risk predictions, 16121 were actually low_risk credits.

![EasyEnsemble.png](https://github.com/Thinguyen23/Thi_M17_SupervisedMachineLearning/blob/master/images/BalancedRandomForest.png)

2. Recall score for high_risk credits is 93/(93+8)=0.92, and for low_risk credits is 16121/(16121+983)=0.94. A high recall is idicative of a small number of False Negatives as compared to True Positives - of the 17104 actual low_risk credits, only 983 were predicted as high_risk credits.

3. Balanced acurracy score is the average accuracy score per class. The balanced accuracy score is calculated as (93/101 + 16121/17104)/2=0.932, which means of all 101 actual high_risk credits, 93 were correctly predicted and of all  17104 actual low_risk credits, 16121 were correctly predicted.

### Recommendation
The model is by far the best to assess credit risk because the balanced accuracy, 0.932, is high. Precision and recall scores are also high for high_risk and low_risk classes.

