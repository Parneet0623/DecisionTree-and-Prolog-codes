# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 18:53:19 2020

@author: parneet
"""

# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus


pima = pd.read_csv('book1.csv')


#split dataset in features and target variable
feature_cols = ['Sore Throat ', 'Fever', 'Swollen Gland', 'Congestion','Headache']
X = pima[feature_cols] # Features
y = pima.label # Target variable

train_features = pima.iloc[:3,:-1]
test_features = pima.iloc[3:,:-1]
train_targets = pima.iloc[:3,-1]
test_targets = pima.iloc[3:,-1]


tree = DecisionTreeClassifier(criterion = 'entropy').fit(train_features,train_targets)

prediction = tree.predict(test_features)

"""
Check the accuracy
"""

print("The prediction accuracy is: ",tree.score(test_features,test_targets)*100,"%")


dot_data = StringIO()
export_graphviz(tree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['Strep throat','Allergy','Cold' ])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())