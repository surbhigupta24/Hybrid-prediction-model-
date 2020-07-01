# Hybrid-prediction-model-
A comprehensive data level investigation of cancer diagnosis on imbalanced data 

Installation: To run the scripts, you need to have installed:

Spyder(Python) 
Python 3.7
Python packages panda
pip install panda
Python packages panda
pip install numpy
pip install keras
pip install tensorflow

You need to have root privileges, an internet connection, and at least 1 GB of free space on your hard disk.
Our scripts were originally developed on a Dell -15JPO9P computer with an Intel Core i7-8550U CPU 1.80GHz processor, with 8 GB of Random-Access Memory (RAM).

Dataset preparation: The data sets used for the analysis are publically available on the website of the University of California Irvine Machine Learning Repository, under its copyright license.
All the data files are shared with the code. Also, few of the data sources are online available. The respective links are available as following:

Download the Cervical cancer (Risk Factors) Data Set file  at the following URL: 
https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29

Download the Mesothelioma’s disease data set Data Set file  at the following URL: 
https://archive.ics.uci.edu/ml/datasets/Mesothelioma%C3%A2%E2%82%AC%E2%84%A2s+disease+data+set+

Download the Breast Cancer Wisconsin (Diagnostic) Data Set file  at the following URL: 
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

Proposed Classification Approach

The Python code for the data balancing techniques is shared in the seperate file:
Data_balance.py

The Python code for the developing the Proposed Stacked Ensemble Classifier is shared in a seperate file:
Ensemble_stacking 

The Python code for the developing the Ensemble of simple Classifier is shared in a seperate file:
Simple_stacking

To execute the Ensemble of simple Classifier on Cervical Cancer Dataset: 
Run Cervical_SS.py

To execute the Ensemble of simple Classifier on Mesothelioma Dataset:
Run Mesothelioma_SS.py

To execute the Ensemble of simple Classifier on Breast Cancer Wisconsin Dataset:
Run Breast_SS.py

To execute the Proposed Stacked Ensemble Classifier on Cervical Cancer Dataset: 
Run Cervical_ES.py

To execute the Proposed Stacked Ensemble Classifier on Mesothelioma Dataset:
Run Mesothelioma_ES.py

To execute the Proposed Stacked Ensemble Classifier on Breast Cancer Wisconsin Dataset:
Run Breast_ES.py

Reference
More information about this project can be found on this paper:
Surbhi Gupta and Manoj K. Gupta "A comprehensive data level investigation of cancer diagnosis on imbalanced data".

Contacts
This sofware was developed by Surbhi Gupta at the School of Computer Science & Engineering, Shri Mata Vaishno Devi University, Sub-Post Office,  Network Centre, Katra, Jammu and Kashmir 182320, India . 
For questions or help, please write to sur7312@gmail.com 