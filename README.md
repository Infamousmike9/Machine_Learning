# Machine Learning

### Student: Michael De Paula


In this project we are creating two notebooks (Credit Risk Ensemble and Credit Risk Resampling) to analyze data related to mortgage lending. Our analysis will use models based on machine learning specifically classification.

We are using the following imports in the beginning of the code:

- import numpy as np
- import pandas as pd
- rom pathlib import Path
- from collections import Counter
- from sklearn.model_selection import train_test_split
- from sklearn.preprocessing import StandardScaler
- from imblearn.over_sampling import RandomOverSampler
- from sklearn.linear_model import LogisticRegression
- from sklearn.metrics import confusion_matrix
- from imblearn.metrics import classification_report_imbalanced
- from sklearn.metrics import balanced_accuracy_score
- from imblearn.over_sampling import SMOTE
- from imblearn.under_sampling import ClusterCentroids
- from imblearn.combine import SMOTEENN
- from imblearn.ensemble import BalancedRandomForestClassifier

The mortgage data being analyzed is saved in a CSV file in the Resources folder under the name LoanStats_2019Q1 which contains loan statistical data that may be used to identify hgh risk and low risk borrowers. 

We will be splitting the data into Test and Train data and setting a targets and feature in order to begin preparing the information for modelling. Within the Ensemble notebook we will be creating a Balanced Random Forest Classifier that will provide us with feature importance scores. 

In our resampling notebook will will use oversampling model methods like the Naive Random Oversampling and Smote Oversampling. We will also be using Undersampling models like ClusterCentroids and finally combining the samples under SMOTEENN. 

