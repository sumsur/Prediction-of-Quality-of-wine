#For prediction we will use:
#1. Random Forest Classifier
#2. Stochastic Gradient Descent Classifier
#3. Support Vector Classifier(SVC)

#For crossvalidation evaluation tecnique we will use:
#1. Grid Search CV
#2. Cross Validation Score

#packages:
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

#data
wine=pd.read_csv(r'C:/Users/tatja/OneDrive/Desktop/Python/redwine.csv')
#head
print(wine.head())
print(wine.info())

#visualization of data
fig, axes = plt.subplots(1, 7)
sns.barplot(x='quality', y='volatile acidity', data=wine, orient='v',  ax=axes[0])

#Here we see that its quite a downing trend in the volatile acidity as we go higher the quality 
#And when we are talking about citric acid:

sns.barplot(x = 'quality', y = 'citric acid', data = wine, orient='v',  ax=axes[1])
#And sugar is standard that is residual

sns.barplot(x = 'quality', y = 'residual sugar', data = wine, orient='v',  ax=axes[2])

#Composition of cloride also go down as we go higher in the quality of wine

sns.barplot(x = 'quality', y = 'chlorides', data = wine, orient='v',  ax=axes[3])


sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = wine, orient='v',  ax=axes[4])

#Alcohol level also goes higher as te quality of wine increases
#THAT IS OF THE MAIN IMPORTANT
sns.barplot(x = 'quality', y = 'alcohol', data = wine, orient='v',  ax=axes[5])
#And sugar is standard that is residual
sns.barplot(x = 'quality', y = 'residual sugar', data = wine, orient='v',  ax=axes[6])

plt.show()

#Preprocessing Data for performing Machine learning algorithms
#Making binary classificaion for the response variable.
#Dividing wine as good and bad by giving the limit for the quality

bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)
#Now lets assign a labels to our quality variable
label_quality = LabelEncoder()

#Bad becomes 0 and good becomes 1 
wine['quality'] = label_quality.fit_transform(wine['quality'])
print(wine['quality'].value_counts())

sns.countplot(wine['quality'])

plt.show()

#Now seperate the dataset as response variable and feature variabes
X = wine.drop('quality', axis = 1)
y = wine['quality']

#Train and Test splitting of data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
#Applying Standard scaling to get optimized result
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Our training and testing data is ready now to perform machine learning algorithm

#RANDOM FOREST CLASSIFIER
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)

#Let's see how our model performed
print(classification_report(y_test, pred_rfc))

#Random forest gives the accuracy of 87%
#Confusion matrix for the random forest classification
print(confusion_matrix(y_test, pred_rfc))

#stichastic gradient decent classifier
sgd = SGDClassifier(penalty=None)
sgd.fit(X_train, y_train)
pred_sgd = sgd.predict(X_test)
print(classification_report(y_test, pred_sgd))
#84% accuracy using stochastic gradient descent classifier
print(confusion_matrix(y_test, pred_sgd))

#SUPPORT VECTOR CLASSIFIER
svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)
print(classification_report(y_test, pred_svc))

#Finding best parameters for our SVC model
param = {
    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
}
grid_svc = GridSearchCV(svc, param_grid=param, scoring='accuracy', cv=10)
grid_svc.fit(X_train, y_train)
#Best parameters for our svc model
grid_svc.best_params_
#Let's run our SVC again with the best parameters.
svc2 = SVC(C = 1.2, gamma =  0.9, kernel= 'rbf')
svc2.fit(X_train, y_train)
pred_svc2 = svc2.predict(X_test)
print(classification_report(y_test, pred_svc2))
#SVC improves from 86% to 90% using Grid Search CV
#Cross Validation Score for random forest and SGD

#Now lets try to do some evaluation for random forest model using cross validation.
rfc_eval = cross_val_score(estimator = rfc, X = X_train, y = y_train, cv = 10)
print(rfc_eval.mean())

#Random forest accuracy increases from 87% to 91 % using cross validation score











