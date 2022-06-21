# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:46:20 2022

Age : Age of the patient

Sex : Sex of the patient

exang: exercise induced angina (1 = yes; 0 = no)

ca: number of major vessels (0-3)

cp : Chest Pain type chest pain type

Value 1: typical angina
Value 2: atypical angina
Value 3: non-anginal pain
Value 4: asymptomatic
trtbps : resting blood pressure (in mm Hg)

chol : cholestoral in mg/dl fetched via BMI sensor

fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)

rest_ecg : resting electrocardiographic results

Value 0: normal
Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
thalach : maximum heart rate achieved

target : 0= less chance of heart attack 1= more chance of heart attack


@author: Khuru
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import pickle


#%%
import scipy.stats as ss
def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


#%% Step 1 Data Loading

#read data
DATA_PATH = os.path.join(os.getcwd(),'heart.csv')

#%% EDA
df = pd.read_csv(DATA_PATH)

#Step 2: Data Inspection
df.info()
df.describe().T
#categorical data: sex,cp,fbs,restcg,exng,slp,caa,thall
#continuos data : age,trtbps,chol,thalachh,oldpeak

con_columns = ['age','trtbps','chol','thalachh','oldpeak']
for con in con_columns:
    plt.figure()
    sns.distplot(df[con])
    plt.show()
    
cater_columns = ['sex','cp','fbs','restecg','exng','slp','caa','thall']
for cat in cater_columns:
    plt.figure()
    sns.countplot(df[cat])
    plt.show()
    
#early hypothesis: little inference can be drawn when analysing

df.describe().T
df.boxplot()
df.columns

cor_mat = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(cor_mat, annot=True)
plt.show()

#%% Step 3) Data Cleaning

#check for duplicated data
df.duplicated().sum()
df[df.duplicated()]

#Remove duplicated data
df = df.drop_duplicates()    
print(df.duplicated().sum())
print(df[df.duplicated()]) 

#Data Imputation by using Simple Imputer
df['thall']=df['thall'].replace(0,np.nan)
df.isna().sum()
df['thall'].fillna(df['thall'].mode()[0],inplace=True)
df.isna().sum()

#Check if there is NaNs
df.isna().sum()  #there is no NaNs 

#%% Feature selection

df_copied = df.columns

for con in con_columns:
    print(con)
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df[con], axis = 1), df['output'])
    print(lr.score(np.expand_dims(df[con], axis = 1), df['output']))


# Categorical vs Categorical
for cat in cater_columns:
    print(cat)
    confussion_mat = pd.crosstab(df[cat], df['output']).to_numpy()   
    print(cramers_corrected_stat(confussion_mat))
    

#Step 5) Preprocessing
X = df.loc[:,['age','trtbps','chol','thalachh','oldpeak','cp','thall']]
y = df['output']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)

#%%pipeline

#LR
step_mms_lr = Pipeline([('mmsscaler',MinMaxScaler()),
                        ('lr',LogisticRegression())])
                       
step_ss_lr = Pipeline([('mmsscaler',StandardScaler()),
                        ('lr',LogisticRegression())])
                       
#RF                      
step_mms_rf = Pipeline([('mmsscaler',MinMaxScaler()),
                        ('rf',RandomForestClassifier())])
                       
step_ss_rf = Pipeline([('mmsscaler',MinMaxScaler()),
                        ('rf',RandomForestClassifier())])    
                      
#tree
step_mms_tree = Pipeline([('mmsscaler',MinMaxScaler()),
                        ('tree',DecisionTreeClassifier())])
                       
step_ss_tree = Pipeline([('mmsscaler',MinMaxScaler()),
                        ('tree',DecisionTreeClassifier())])
                                               
#knn
step_mms_knn = Pipeline([('mmsscaler',MinMaxScaler()),
                        ('knn',KNeighborsClassifier())])
                       
step_ss_knn = Pipeline([('mmsscaler',StandardScaler()),
                        ('knn',KNeighborsClassifier())])

#%%
#pipelines    
pipelines = [step_mms_lr,step_ss_lr,step_mms_rf,step_ss_rf,
             step_mms_tree,step_ss_tree,step_mms_knn,step_ss_knn]

for pipe in pipelines:
    pipe.fit(X_train,y_train)
    
best_accuracy = 0

for i, model in enumerate(pipelines):
    print(model.score(X_test,y_test))
    if model.score(X_test,y_test) > best_accuracy:
        best_accuracy = model.score(X_test,y_test)
        best_pipeline = model
        
#%% this is to fine tune the model
# Steps for Random Forest
step_rf = [('Standard Scaler', StandardScaler()),
           ('RandomForestClassifier',RandomForestClassifier(random_state=123))]

pipeline_rf=Pipeline(step_rf)

#number of trees
grid_param = [{'RandomForestClassifier':[RandomForestClassifier()],
               'RandomForestClassifier__n_estimators':[10,100,1000],
               'RandomForestClassifier__max_depth':[None,5,15]}]



gridsearch = GridSearchCV(pipeline_rf,grid_param,cv=5,verbose=1,n_jobs=-1)
best_model = gridsearch.fit(X_train,y_train)
print(best_model.score(X_test,y_test))
print(best_model.best_index_)
print(best_model.best_params_)

#%% MODEL SAVING 
BEST_MODEL_PATH = os.path.join(os.getcwd(),'best_model.pkl')
with open(BEST_MODEL_PATH,'wb') as file:
    pickle.dump(best_model,file)
    
   
    
BEST_PIPE_PATH = os.path.join(os.getcwd(),'best_pipe.pkl')
with open(BEST_PIPE_PATH,'wb') as file:
    pickle.dump(best_model,file)

 

