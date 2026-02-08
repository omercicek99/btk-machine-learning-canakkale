import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns

dir_path=os.path.dirname(os.path.realpath('__file__'))
df=pd.read_csv(dir_path+"/heart_failure.csv")
print(df.head(5))
print(df.columns)

#x = data.drop('NObeyesdad', axis=1)

X = df.drop(["DEATH_EVENT"], axis=1)
y = df["DEATH_EVENT"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)





knn=KNeighborsClassifier()
n_neighbors=list(range(3,99,2))
weights=['unifrom','distance']
algorithm=['ball_three','kd_tree', 'brute','auto']
metric=['euclieadn','minkowski','manhasttan']
hyperparamters=dict(n_neighbors=n_neighbors,weights=weights,algorithm=algorithm,metric=metric)
clf=GridSearchCV(knn,hyperparamters)
best_model=clf.fit(X_train,y_train)

n_neighbors=best_model.best_estimator_.get_params()['n_neighbors']
weights=best_model.best_estimator_.get_params()['weights']
algorithm=best_model.best_estimator_.get_params()['algorithm']
metric=best_model.best_estimator_.get_params()['metric']
print(n_neighbors,weights,algorithm,metric)

import joblib
model_filename='knn_model.joblib'
joblib.dump(knn,model_filename)
print("model kaydedildi:" + model_filename)
