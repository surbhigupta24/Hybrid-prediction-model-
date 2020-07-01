# -*- coding: utf-8 -*-
"""
Created on Tue May 26 17:46:54 2020

@author: SURBHI
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 26 17:41:49 2020

@author: SURBHI
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 23:51:38 2019
@author: SURBHI"""


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

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import statistics
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef


data = pd.read_csv('cervical.csv')
des = data.describe()
inf = data.info()

null = data.isna().sum()

null = {}
c = 0
n = 0
for i in data.columns:
    for j in data[i]:
        if j == '?':
            c = c+1
    null[i] = c
    c = 0
    
a = {key: val for key, val in null.items() if val > 0}

for i in data.columns:
    for j in data.index: 
        if data[i][j] == '?':
            data[i][j] = np.nan
        else:
            pass
        
data.isna().sum()
data = data.drop(columns = ['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'])
data = data.apply(pd.to_numeric, errors='coerce')
d = {}
for i in data:
    d[i] = data[i].nunique()
    
from missingpy import KNNImputer 
cols = list(data)

data1 = pd.DataFrame(KNNImputer().fit_transform(data))
data1.columns = cols

data1.isna().sum()        

data1 = data1.dropna()
data1.isna().sum()
data1['Biopsy'].value_counts()
data1['Hinselmann'].value_counts()
data1['Citology'].value_counts()
data1['Schiller'].value_counts()

data1 = data1.drop(columns = ['Hinselmann', 'Citology','Biopsy'])
X = data1.iloc[:, :-1].values
y = data1.iloc[:, -1].values

################################################ 
#                  Normalization                #
################################################  
 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#################### Simple Ensemble and stacking ##########################

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier  
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import statistics
ran = []
xg = []
gb1 = []
Voting = []
Neural_Stacking = []
Bayesian_Stacking = []
Logistic_Stacking = []
Boosted_Stacking = []
   

def Simple_Stacking(X, y):
    
    ############ SVC

    cv = KFold(n_splits=10, random_state=20, shuffle=True)
    for train_index, test_index in cv.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    
        DT = DecisionTreeClassifier( criterion = "entropy", random_state = 42,)
        DT.fit(X_train, y_train)
        y_predr = DT.predict(X_test)
        ran.append(metrics.accuracy_score(y_test, y_predr))
  
    ########### KNN
    
        knn = KNeighborsClassifier() 
        knn.fit(X_train, y_train)
        y_predx = knn.predict(X_test)
#        print('XGBClassifier 1 - ' + str(metrics.accuracy_score(y_test, y_predx)))
        xg.append(metrics.accuracy_score(y_test, y_predx))
        
    ############ GNB
    
        gnb= GaussianNB()
        gnb.fit(X_train, y_train)
        y_preds = gnb.predict(X_test)
        gb1.append(metrics.accuracy_score(y_test, y_preds))
               
    ############### Voting Stacking
    
        voting_claccuracy = VotingClassifier(
            estimators=[('DT', DT), ('knn', knn), ('gnb', gnb)],
            voting='hard', weights = [3,3,2])
        voting_claccuracy.fit(X_train, y_train)
        y_pred1 = voting_claccuracy.predict(X_test)
            
        sig_claccuracy = CalibratedClassifierCV(DT, method="sigmoid")
        sig_clf2 = CalibratedClassifierCV(knn, method="sigmoid")
        sig_clf3 = CalibratedClassifierCV(gnb, method="sigmoid")
        sclaccuracy = StackingCVClassifier(classifiers = [sig_claccuracy, sig_clf2, sig_clf3],
                            shuffle = True,
                            meta_classifier = LogisticRegression())
        sclaccuracy.fit(X_train, y_train)  
        y_pred2 = sclaccuracy.predict(X_test)
        
        nb = XGBClassifier()
        sclf2 = StackingClassifier(classifiers=[sig_claccuracy, sig_clf2, sig_clf3], meta_classifier=nb,use_probas=False)
        sclf2.fit(X_train, y_train)  
        y_pred3 = sclf2.predict(X_test)
        
        lr = LogisticRegression()
        sclf3 = StackingClassifier(classifiers=[sig_claccuracy, sig_clf2, sig_clf3], meta_classifier=lr,use_probas=False)
        sclf3.fit(X_train, y_train)  
        y_pred4 = sclf3.predict(X_test)
        
        gb = GradientBoostingClassifier()
        sclf4 = StackingClassifier(classifiers=[sig_claccuracy, sig_clf2, sig_clf3], meta_classifier=gb,use_probas=False)
        sclf4.fit(X_train, y_train)  
        y_pred5 = sclf4.predict(X_test)
        
        Voting.append(metrics.accuracy_score(y_pred1, y_test))
        Neural_Stacking.append(metrics.accuracy_score(y_pred2, y_test))
        Bayesian_Stacking.append(metrics.accuracy_score(y_pred3, y_test))
        Logistic_Stacking.append(metrics.accuracy_score(y_pred4, y_test))
        Boosted_Stacking.append(metrics.accuracy_score(y_pred5, y_test))
        
    ran_mean = statistics.mean(ran)
    print('DT: %.2f' %ran_mean)
    xg_mean = statistics.mean(xg)
    print('KNN: %.2f' %xg_mean)
    gb1_mean = statistics.mean(gb1)
    print('GNB: %.2F' %gb1_mean)
    Voting_mean = statistics.mean(Voting)
    print('VOTING: %.2f' %Voting_mean)
    Neural_Stacking_mean = statistics.mean(Neural_Stacking)
    print('NEURAL: %.2f' %Neural_Stacking_mean)
    Bayesian_Stacking_mean = statistics.mean(Bayesian_Stacking)
    print('XGBoosting: %.2f' %Bayesian_Stacking_mean)
    Logistic_Stacking_mean = statistics.mean(Logistic_Stacking)
    print('LOGISTIC %.2f' %Logistic_Stacking_mean)
    Boosted_Stacking_mean = statistics.mean(Boosted_Stacking)
    print('BOOSTED: %.2f' %Boosted_Stacking_mean)
    



########### Without Balancing ################

Simple_Stacking(X, y)

################################################ 
#                      Balancing               #
################################################


################################### Under Sampling Technique#######################


# 'UnderSampler'
US = RandomUnderSampler()
usx, usy = US.fit_sample(X, y)
Simple_Stacking(usx, usy)

# 'Tomek links'
TL = TomekLinks()
tlx, tly = TL.fit_sample(X, y)
Simple_Stacking(tlx, tly)

################################### Combined Sampling Technique#######################

########### 'SMOTE Tomek links'

STK = SMOTETomek()
stkx, stky = STK.fit_sample(X, y)
Simple_Stacking(stkx, stky)

########### 'SMOTE ENN'

SENN = SMOTEENN()
ennx, enny = SENN.fit_sample(X, y)
Simple_Stacking(ennx,enny)


################################### Over Sampling Technique#######################

########## 'Random over-sampling'
OS = RandomOverSampler()
osx, osy = OS.fit_sample(X, y)
Simple_Stacking(osx, osy)

########### 'SMOTE bordeline 1'
bsmote1 = SMOTE(kind='borderline1')
bs1x, bs1y = bsmote1.fit_sample(X, y)
Simple_Stacking(bs1x, bs1y)

########### 'SMOTE bordeline 2'
bsmote2 = SMOTE(kind='borderline2')
bs2x, bs2y = bsmote2.fit_sample(X, y)
Simple_Stacking(bs2x, bs2y)

########### 'SMOTE SVM'
svmsmote = SMOTE(kind='svm')
svsx, svsy = svmsmote.fit_sample(X, y)
Simple_Stacking(svsx, svsy)

############ GuassianSMOTE

gs = sv.Gaussian_SMOTE(n_neighbors=5, random_state=30)
X_gs, y_gs = gs.sample(X, y)
Simple_Stacking(X_gs, y_gs)


