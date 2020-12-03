from sklearn.tree import DecisionTreeClassifier
import pandas as pd

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split




pima = pd.read_csv('Book1.csv')
print(pima)

x=pima.iloc[:,:5]
y=pima.iloc[:,5]
xTr,xTe,yTr,yTe=train_test_split(x,y,test_size=0.2)

dtc=DecisionTreeClassifier()
dtc.fit(xTr,yTr)
yPred=dtc.predict(xTe)
print(confusion_matrix(yTe,yPred))
print(accuracy_score(yTe,yPred))
print(classification_report(yTe,yPred))
