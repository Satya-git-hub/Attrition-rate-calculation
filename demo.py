import pandas as pd

data = pd.read_csv("D:/IT DOCxxx/datasets/attrition.csv")
#data.head()

#Creating new/derived predictors (e.g. Age group) for analysis

data['Age'].mean()
dummy=[]
for i in data['Age']:
    if i<36.92:
        dummy.append('Juniors')
    else:
        dummy.append('Seniors')

cat_data=data[["Attrition","BusinessTravel","Department","Education","EducationField","EmployeeCount","EnvironmentSatisfaction","Gender","JobInvolvement","JobLevel","JobRole","JobSatisfaction","MaritalStatus","Over18","OverTime","PerformanceRating","RelationshipSatisfaction","StandardHours","StockOptionLevel","TrainingTimesLastYear","WorkLifeBalance","NumCompaniesWorked","PercentSalaryHike"]]
data['AgeGroup']=dummy	
cont_data=data[["Age","DailyRate","DistanceFromHome","EmployeeNumber","HourlyRate","MonthlyIncome","MonthlyRate","TotalWorkingYears","YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager"]]

cont_data.head()

import matplotlib.pyplot as plt
plt.scatter(cont_data["Age"],cont_data["MonthlyIncome"])

import seaborn as sns
sns.pairplot(cont_data)

#Settign the predictors and the target 
y=cat_data[["Attrition"]]
x=data.drop("Attrition",axis=1)

from sklearn.preprocessing import LabelEncoder as LaEn
le=LaEn()
x=x.apply(le.fit_transform)

from sklearn.model_selection import train_test_split as tts
xtrain, xtest, ytrain,ytest = tts(x,y,test_size=0.3,random_state=13)

from sklearn.metrics import accuracy_score as accuracy
################## Model Preparation ##################

#Random Forest
from sklearn.ensemble import RandomForestClassifier as rfc
y=data[["Attrition"]]
x=data.drop("Attrition",axis=1)

from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder()
x=x.apply(le.fit_transform)

from sklearn.model_selection import train_test_split as tts 
xtrain,xtest,ytrain,ytest=tts(x,y,test_size=0.3,random_state=13)


model=rfc()
model.fit(xtrain,ytrain)
pred=model.predict(xtest)

accuracy(ytest,pred)
#Out[61]: 0.8503401360544217


model.feature_importances_

dick={}
for i in range (0,len(model.feature_importances_)):
    if model.feature_importances_[i]>0.02:
        dick[x.columns[i]]=model.feature_importances_[i]

refined_x=data[dick.keys()]
refined_x=refined_x.apply(le.fit_transform)

xtrain,xtest,ytrain,ytest=tts(refined_x,y,test_size=0.3,random_state=13)

param={'criterion':['gini','entropy'],'max_depth':range(1,10),'max_leaf_nodes':range(2,10)}

from sklearn.model_selection import GridSearchCV as gscv
cv=gscv(model,param,n_jobs=-1,scoring='accuracy',cv=5)
cv_model=cv.fit(xtrain,ytrain)

cv_model.best_params_
#Out[79]: {'criterion': 'entropy', 'max_depth': 5, 'max_leaf_nodes': 9}

rf_model=rfc(criterion= 'entropy', max_depth=5, max_leaf_nodes=9,random_state=13)

rf_model.fit(xtrain,ytrain)
final_pred=rf_model.predict(xtest)

accuracy(ytest,final_pred)
#Out[88]: 0.8503401360544217

#########Boosting#########
#Gradient boosting 

#Adaptive boosting 

#xGradient boosting 
from xgboost import  XGBClassifier as xgb
classifier=xgb()
classifier.fit(xtrain,ytrain)

xgb_pred=classifier.predict(xtest)
accuracy(ytest,xgb_pred)

#Out[93]: 0.8616780045351474

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,xgb_pred)
cm

param={'max_depth':range(1,20),'n_estimators':range(1,10),'booster':['dart','gbtree'],'importance_type':['gain', 'weight', 'cover', 'total_gain','total_cover']}

from sklearn.model_selection import GridSearchCV as gscv
cv=gscv(classifier,param,n_jobs=-1,cv=10,scoring='accuracy')
cv_model=cv.fit(xtrain,ytrain)
cv_model.best_params_

xg=xgb(max_depth=3,n_estimators=8,booster='dart',
       importance_type='gain',random_state=13,n_jobs=-1)

xg.fit(xtrain,ytrain)
xg_pred=xg.predict(xtest)
accuracy(ytest,xg_pred)

#Out[102]: 0.8594104308390023
