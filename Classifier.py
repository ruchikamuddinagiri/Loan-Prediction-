import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import math
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import log_loss

#get the training dataset
get_ipython().system('wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')

#Some visualisation code
#df['loan_status'].value_counts()
#df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
#df.groupby(['education'])['loan_status'].value_counts(normalize=True)

#load the training dataset
df = pd.read_csv('loan_train.csv')

#pre-process the data
df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)

#define Feature set
Feature = df[['Principal','terms','age','Gender']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)

#create feature set and labels
x = Feature
x[0:5]
y = df['loan_status'].values
y[0:5]

#create two dictionaries to hold the evaluation metrics
trainscores={}
testscores={}

#split the given training dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y)

#normalize the data
x= preprocessing.StandardScaler().fit(x).transform(x.astype(float))


#start training and testing on the dataset
#KNN
#run the loop to find the best k
Ks = int(math.sqrt(x.shape[0]))
j_acc = np.zeros((Ks-1))
f_acc = np.zeros((Ks-1))
#ConfustionMx = [];
for n in range(1,Ks):
    #Train Model and Predict 
    neigh = KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)
    #predict
    y_hat=neigh.predict(x_test)
    #jaccard similarity score
    j_acc[n-1] = jaccard_similarity_score(y_test, y_hat)
    #f1score
    f_acc[n-1]=f1_score(y_test, y_hat, average='weighted',labels=np.unique(y_hat))
trainscores['KNN-jaccard']=f_acc.max()
trainscores['KNN-F1']=j_acc.max()


#Decision Tree

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree.fit(x_train,y_train)
y__hat = drugTree.predict(x_test)
trainscores['Decision tree-Jaccard']= jaccard_similarity_score(y_test, y__hat)
trainscores['Decision tree- f1 score']=f1_score(y_test, y__hat, average='weighted', labels=np.unique(y__hat))

#Support Vector Machine

clf = svm.SVC(kernel='rbf', gamma="auto")
clf.fit(x_train, y_train)
y___hat = clf.predict(x_test)
trainscores['svm -jaccard']=jaccard_similarity_score(y_test, y___hat)
trainscores['svm-f1 score']=f1_score(y_test, y___hat, average='weighted', labels=np.unique(y___hat))


#Logistic Regression
LR = LogisticRegression(C=0.000001, solver='saga').fit(x_train,y_train)
y____hat = LR.predict(x_test)
yhat_prob = LR.predict_proba(x_test)
trainscores['logistic regr- jaccard']=jaccard_similarity_score(y_test, y_hat)
trainscores['logistic regr- f1 score']=f1_score(y_test, y_hat, average='weighted', labels=np.unique(y_hat))
trainscores['logistic regr- log loss']=log_loss(y_test, yhat_prob)

#####################################################
#Out-of-sample prediction and model evaluation
#####################################################


#get the test dataset for prediction and evaluation of the models previously trained
get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


#Prepare Test set for evaluation
test_df = pd.read_csv('loan_test.csv')
test_df.head()
test_df['due_date']=pd.to_datetime(test_df['due_date'])
test_df['effective_date']=pd.to_datetime(test_df['effective_date'])
test_df.head()
test_df['Gender'].replace(to_replace=['male', 'female'], value=[0,1], inplace=True)
test_Feature=test_df[['Principal', 'terms', 'age', 'Gender']]
test_Feature= pd.concat([test_Feature, pd.get_dummies(test_df['education'])], axis=1)
test_Feature.drop(['Master or Above'], axis=1, inplace=True)

x=test_Feature
y=test_df['loan_status'].values

#Logistic regression prediction
l_y_hat = LR.predict(x)
l_yhat_prob = LR.predict_proba(x)
testscores['logistic regr- jaccard']=jaccard_similarity_score(y_test, y_hat)
testscores['logistic regr- f1 score']=f1_score(y_test, y_hat, average='weighted', labels=np.unique(y_hat))
testscores['logistic regr- log loss']=log_loss(y_test, yhat_prob)

#Support Vector Machine prediction
s_y_hat = clf.predict(x)
testscores['svm -jaccard']=jaccard_similarity_score(y, s_y_hat)
testscores['svm-f1 score']=f1_score(y, s_y_hat, average='weighted', labels=np.unique(s_y_hat))

#Decision tree prediction
d_y_hat = drugTree.predict(x)
testscores['decision tree -jaccard']=jaccard_similarity_score(y, d_y_hat)
testscores['decision tree -f1 score']=f1_score(y, d_y_hat, average='weighted', labels=np.unique(d_y_hat))

#K-Nearest Neighbors prediction
Ks = int(math.sqrt(x.shape[0]))
j_acc = np.zeros((Ks-1))
f_acc = np.zeros((Ks-1))
#ConfustionMx = [];
for n in range(1,Ks):
    
    #predict
    y_hat=neigh.predict(x)
    #jaccard similarity score
    j_acc[n-1] = jaccard_similarity_score(y, y_hat)
    #f1score
    f_acc[n-1]=f1_score(y, y_hat, average='weighted',labels=np.unique(y_hat))
print("F1 score score for KNN is ", f_acc.max())
print("Jaccard score score for KNN is ", j_acc.max())
testscores['KNN-jaccard']=f_acc.max()
testscores['KNN-F1']=j_acc.max()

#print the evaluation metrics
print(trainscores)
print(testscores)