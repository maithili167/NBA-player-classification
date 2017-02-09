
#importing all the required modules for the classifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm, preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

#Read csv file into a dataframe
nba=pd.read_csv("NBAstats.csv")

#Class attribute
clas_column='Pos'
#features selected after performing recursive feature elimination and feature subset selection
feature_colums=['FG%','eFG%','FT%', 'ORB', 'DRB','AST', 'STL','BLK','TOV','PS/G']
nba_feature=nba[feature_colums]

#Removing features with low variance
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(nba_feature)

#Below steps is perfromed for attribute transformation through standardization
nba_feature=StandardScaler(with_mean=True, with_std=True).fit_transform(nba_feature)
nba_class=nba[clas_column]

#Dataset is split into train(75%) and test(25%) set
train_feature,test_feature,train_class,test_class=train_test_split(
    nba_feature,nba_class,stratify=nba_class,train_size=0.75,test_size=0.25)

#Model is trained using training dataset received after performing above step
svc = svm.SVC(C=1000.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma=0.001, kernel='rbf',
  max_iter=-1, probability=False,  shrinking=True,
  tol=0.001, verbose=False)
svc.fit(train_feature,train_class)

#score function of the classifier returns the accuracy of the test set
# print("Training accuracy:{:.3f}".format(svc.score(train_feature,train_class)))
print("Testing accuracy:{:.3f}".format(svc.score(test_feature,test_class)))


prediction=svc.predict(test_feature)
'''Predicted class values of the test data from the above step is used to
construct the confusion matrix'''
print("Confusion matrix:")
print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))

'''10 fold Stratified cross validation performed on the entire dataset using
appropriate parameters
'''
scores=cross_val_score(svc,nba_feature,nba_class,cv=StratifiedKFold(n_splits=10))
#Below step returns score by each fold in 10 fold cross validation
print("Cross validation scores:",scores)
#scores over 10 fold cross validation is averaged in the below step
print("Average cross validation score:{:.2f}".format(scores.mean()))


