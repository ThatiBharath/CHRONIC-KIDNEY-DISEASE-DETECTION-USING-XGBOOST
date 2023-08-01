import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
sns.set(color_codes=True)
Data=pd.read_csv('data.csv')
Data.info()
np.size(Data)
np.shape(Data)
Data.describe().T
Data.isnull().sum()
Data['AGE']=Data['AGE'].fillna(Data['AGE'].mean())
Data['WEIGHT']=Data['WEIGHT'].fillna(Data['WEIGHT'].mean())
Data['SG']=Data['SG'].fillna(Data['SG'].mean())
Data['Alb']=Data['Alb'].fillna(Data['Alb'].mean())
Data['eGFR']=Data['eGFR'].fillna(Data['eGFR'].mean())
Data['Na+’]=Data['Na+'].fillna(Data['Na+'].mean())
Data['K+']=Data['K+'].fillna(Data['K+'].mean())
Data['BUN']=Data['BUN'].fillna(Data['BUN'].mean())
Data['Creat']=Data['Creat'].fillna(Data['Creat'].mean())
Data['Rbc']=Data['Rbc'].fillna(Data['Rbc'].mean())
Data['Wbc']=Data['Wbc'].fillna(Data['Wbc'].mean())
sns.distplot(Data.AGE,bins=30);
sns.distplot(Data['Na+'],bins=30);
sns.kdeplot(Data.WEIGHT,shade=True)
sns.kdeplot(Data.Alb,shade=True)
fig=plt.gcf();
fig.set_size_inches(20,6);
sns.swarmplot(x='Bp',y='eGFR',hue='CLASS',data=Data);
sns.swarmplot(x='Appetite',y='eGFR',hue='CLASS',data=Data);
sns.swarmplot(x='HTN',y='eGFR',hue='CLASS',data=Data);
sns.swarmplot(x='DM',y='eGFR',hue='CLASS',data=Data);
sns.swarmplot(x='Anemia',y='eGFR',hue='CLASS',data=Data);
sns.swarmplot(x='BUN',y='eGFR',hue='CLASS',data=Data);
sns.swarmplot(x='Creat',y='eGFR',hue='CLASS',data=Data);
Data['combined']=Data.CLASS.apply(lambda x: 'Mild-mod CKD' if x=='Mild-mod CKD' else 
'ESRD|Severe CKD')
sns.violinplot(x='Na+',y='eGFR',hue='combined',split=True,inner='quartile',data=Data);
sns.violinplot(x='Alb',y='eGFR',hue='combined',split=True,inner='quartile',data=Data);
sns.violinplot(x='Wbc',y='eGFR',hue='combined',split=True,inner='quartile',data=Data);
sns.kdeplot(Data[Data.CLASS=='Severe CKD'].Wbc,shade=True);
sns.kdeplot(Data[Data.CLASS=='ESRD'].Wbc,shade=True);
sns.kdeplot(Data[Data.CLASS=='Mild-mod CKD'].Wbc,shade=True);
plt.legend(title='CLASS',labels=['Server CKD','ESRD','Mild-mod CKD']);
sns.kdeplot(Data[Data.CLASS=='Severe CKD']['K+'],shade=True);
sns.kdeplot(Data[Data.CLASS=='ESRD']['K+'],shade=True);
sns.kdeplot(Data[Data.CLASS=='Mild-mod CKD']['K+'],shade=True);
plt.legend(title='CLASS',labels=['Server CKD','ESRD','Mild-mod CKD']);
sns.kdeplot(Data[Data.CLASS=='Severe CKD'].eGFR,shade=True);
sns.kdeplot(Data[Data.CLASS=='ESRD'].eGFR,shade=True);
sns.kdeplot(Data[Data.CLASS=='Mild-mod CKD'].eGFR,shade=True);
plt.legend(title='CLASS',labels=['Server CKD','ESRD','Mild-mod CKD']);
sns.kdeplot(Data[Data.CLASS=='Severe CKD'].SG,shade=True);
sns.kdeplot(Data[Data.CLASS=='ESRD'].SG,shade=True);
sns.kdeplot(Data[Data.CLASS=='Mild-mod CKD'].SG,shade=True);
plt.legend(title='CLASS',labels=['Server CKD','ESRD','Mild-mod CKD']);
D=sns.FacetGrid(Data,row='CLASS');
D.map(sns.violinplot,'Rbc');
sns.pairplot(Data,hue='CLASS')
fig=plt.gcf();
fig.set_size_inches(17,8);
correlation=Data.corr()
sns.heatmap(abs(correlation), annot=True, cmap='coolwarm')
