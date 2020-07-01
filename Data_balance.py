# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 14:45:43 2020

@author: SURBHI
"""


import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from mlxtend.classifier import StackingCVClassifier 
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss, CondensedNearestNeighbour, OneSidedSelection,NeighbourhoodCleaningRule, TomekLinks, ClusterCentroids
from imblearn.combine import SMOTETomek, SMOTEENN
import smote_variants as sv
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import KFold
sns.set()
import warnings
warnings.simplefilter('ignore')

################################################ 
#                      Balancing               #
################################################


################################### Under Sampling Technique#######################


# 'UnderSampler'
US = RandomUnderSampler()
usx, usy = US.fit_sample(X, y)
Ensemble_and_Stacking(usx, usy)

# 'Tomek links'
TL = TomekLinks()
tlx, tly = TL.fit_sample(X, y)
Ensemble_and_Stacking(tlx, tly)


################################### Combined Sampling Technique#######################

########### 'SMOTE Tomek links'
STK = SMOTETomek()
stkx, stky = STK.fit_sample(X, y)
Ensemble_and_Stacking(stkx, stky)

########### 'SMOTE ENN'

SENN = SMOTEENN()
ennx, enny = SENN.fit_sample(X, y)
Ensemble_and_Stacking(ennx, enny)

################################### Over Sampling Technique#######################

########## 'Random over-sampling'
OS = RandomOverSampler()
osx, osy = OS.fit_sample(X, y)
Ensemble_and_Stacking(osx, osy)

########### 'SMOTE bordeline 1'
bsmote1 = SMOTE(kind='borderline1')
bs1x, bs1y = bsmote1.fit_sample(X, y)
Ensemble_and_Stacking(bs1x, bs1y)

########### 'SMOTE bordeline 2'
bsmote2 = SMOTE(kind='borderline2')
bs2x, bs2y = bsmote2.fit_sample(X, y)
Ensemble_and_Stacking(bs2x, bs2y)

########### 'SMOTE SVM'
svmsmote = SMOTE(kind='svm')
svsx, svsy = svmsmote.fit_sample(X, y)
Ensemble_and_Stacking(svsx, svsy)

############ GuassianSMOTE

gs = sv.Gaussian_SMOTE(n_neighbors=5, random_state=30)
X_gs, y_gs = gs.sample(X, y)
Ensemble_and_Stacking(X_gs, y_gs)


