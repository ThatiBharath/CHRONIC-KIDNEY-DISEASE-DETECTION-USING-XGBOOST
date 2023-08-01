from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
import joblib
df=pd.read_csv('data.csv')
le=LabelEncoder()
datale=df
datale.CLASS=le.fit_transform(datale.CLASS)
datale.HTN=le.fit_transform(datale.HTN)
datale.Appetite=le.fit_transform(datale.Appetite)
datale.Anemia=le.fit_transform(datale.Anemia)
datale.DM=le.fit_transform(datale.DM)
datale=datale.fillna(df.mean().iloc[0])
datale.drop(['ID','Bp','Pedal Edema'],axis=1)
x=datale[['AGE','WEIGHT','SG','Alb','eGFR','Na+','K+','BUN','Creat','Rbc','Wbc','Appetite','Ane
mia','HTN','DM',]].values
y=datale.CLASS
ohe=OneHotEncoder()
ohe.fit_transform(x).toarray()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
LOG=LogisticRegression()
LOG.fit(x_train,y_train)
LOG.score(x_test,y_test)
y_predicted=LOG.predict(x_test)
error=confusion_matrix(y_test,y_predicted)
plt.figure(figsize=(10,7))
sn.heatmap(error,annot=True)
plt.xlabel('predicted')
plt.ylabel('Truth')
DT=tree.DecisionTreeClassifier()
DT.fit(x_train,y_train)
DT.score(x_test,y_test)
y_predicted=DT.predict(x_test)
error=confusion_matrix(y_test,y_predicted)
plt.figure(figsize=(10,7))
sn.heatmap(error,annot=True)
plt.xlabel('predicted')
plt.ylabel('Truth')
RF=RandomForestClassifier()
RF.fit(x_train,y_train)
RF.score(x_test,y_test)
y_predicted=RF.predict(x_test)
error=confusion_matrix(y_test,y_predicted)
plt.figure(figsize=(10,7))
sn.heatmap(error,annot=True)
plt.xlabel('predicted')
plt.ylabel('Truth')
XG= XGBClassifier()
XG.fit(x_train, y_train)
y_predicted = XG.predict(x_test)
XG.score(x_test,y_test)
error=confusion_matrix(y_test,y_predicted)
plt.figure(figsize=(10,7))
sn.heatmap(error,annot=True)
plt.xlabel('predicted')
plt.ylabel('Truth')
XG.save_model("model.h4")
joblib.dump(RF, "model.h3")
joblib.dump(DT,'model.h2')
joblib.dump(LOG,'model.h1')
import PROJECT_TRAINING
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
XG2=XGBClassifier()
XG2.load_model("model.h4")
RF=RandomForestClassifier()
RF2=joblib.load("model.h3")
DT2=tree.DecisionTreeClassifier()
DT2=joblib.load("model.h2")
LOG2=LogisticRegression()
LOG2=joblib.load("model.h1")