from array import array
from enum import auto
from re import sub
from turtle import backward
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from genetic_selection import GeneticSelectionCV
from sklearn.tree import DecisionTreeClassifier


# function for getting all possible combinations of a list
from itertools import chain, combinations
def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))

# function for printing each component of confusion matrix
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

data_path = 'cad_dset.csv'
data = pd.read_csv(data_path)
dataframe = pd.DataFrame(data.values, columns=data.columns)
dataframe['CAD'] = data.CAD
x = dataframe.drop(['ID','female','CNN_Healthy','CNN_CAD','Doctor: CAD','HEALTHY','CAD'], axis=1) # Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
x_nodoc = dataframe.drop(['ID','female','CNN_Healthy','CNN_CAD','Doctor: CAD', 'Doctor: Healthy','HEALTHY','CAD'], axis=1) # Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
y = dataframe['CAD'].astype(int)

# ml algorithms initialization
sv = svm.SVC(kernel='rbf')
dt = DecisionTreeClassifier()
rndF = RandomForestClassifier(max_depth=None, random_state=0, n_estimators=60) #TODO n_estimators=80 when testing with doctor, 60 w/o doctor
ada = AdaBoostClassifier(n_estimators=30, random_state=0) #TODO n_estimators=150 when testing with doctor, 30 w/o doctor
knn = KNeighborsClassifier(n_neighbors=20) #TODO n_neighbors=13 when testing with doctor, 20 w/o doctor

# doc/no_doc parameterization
sel_alg = rndF
x = x_nodoc #TODO comment out when not testing with doctor
X = x_nodoc #TODO set to x when testing with doctor or x_nodoc when NOT testing with doctor


#############################################
#### Genetic Algorithm Feature Selection ####
#############################################

for i in range (0,3):
    print("run no ", i, ":")
    selector = GeneticSelectionCV(
        estimator=sel_alg,
        cv=10,
        verbose=2,
        scoring="accuracy", 
        max_features=26, #TODO change to 27 when testing with doctor, 26 without
        n_population=100,
        crossover_proba=0.8,
        mutation_proba=0.8,
        n_generations=200,
        crossover_independent_proba=0.8,
        mutation_independent_proba=0.4,
        tournament_size=5,
        n_gen_no_change=60,
        caching=True,
        n_jobs=-1)
    selector = selector.fit(x, y)
    n_yhat = selector.predict(x)
    sel_features = x.columns[selector.support_]
    print("Genetic Feature Selection:", x.columns[selector.support_])
    print("Genetic Accuracy Score: ", selector.score(x, y))
    print("Testing Accuracy: ", metrics.accuracy_score(y, n_yhat))

sel_features = x

##############
### CV-10 ####
##############
for feature in x.columns:
    if feature in sel_features:
        pass
    else:
        X = X.drop(feature, axis=1)

sel = sel_alg.fit(X, y)
n_yhat = cross_val_predict(sel, X, y, cv=10)

print("cv-10 accuracy: ", cross_val_score(sel, X, y, scoring='accuracy', cv = 10).mean() * 100)
print("cv-10 accuracy STD: ", cross_val_score(sel_alg, X, y, scoring='accuracy', cv = 10).std() * 100)
print("f1_score: ", cross_val_score(sel_alg, X, y, scoring='f1', cv = 10).mean() * 100)
print("f1_score STD: ", cross_val_score(sel_alg, X, y, scoring='f1', cv = 10).std() * 100)
print("jaccard_score: ", cross_val_score(sel_alg, X, y, scoring='jaccard', cv = 10).mean() * 100)
print("jaccard_score STD: ", cross_val_score(sel_alg, X, y, scoring='jaccard', cv = 10).std() * 100)
scoring = {
    'sensitivity': metrics.make_scorer(metrics.recall_score),
    'specificity': metrics.make_scorer(metrics.recall_score,pos_label=0)
}
print("sensitivity: ", cross_val_score(sel_alg, X, y, scoring=scoring['sensitivity'], cv = 10).mean() * 100)
print("sensitivity STD: ", cross_val_score(sel_alg, X, y, scoring=scoring['sensitivity'], cv = 10).std() * 100)
print("specificity: ", cross_val_score(sel_alg, X, y, scoring=scoring['specificity'], cv = 10).mean() * 100)
print("specificity STD: ", cross_val_score(sel_alg, X, y, scoring=scoring['specificity'], cv = 10).std() * 100)

print("confusion matrix:\n", metrics.confusion_matrix(y, n_yhat, labels=[0,1]))
print("TP/FP/TN/FN: ", perf_measure(y, n_yhat))

# # By running the following loop we found out knn algorithm  gives best results for n=13
# # best features: ['known CAD', 'previous AMI', 'previous CABG', 'Diabetes', 'Smoking', 'Arterial Hypertension', 
# # 'Dislipidemia', 'Angiopathy', 'ASYMPTOMATIC', 'ATYPICAL SYMPTOMS', 'ANGINA LIKE', 'male', 'Overweight', '40b50', 
# # 'o60', 'Doctor: Healthy']
# best_acc=0
# for n in range (40,1,-1):
#     knn = KNeighborsClassifier(n_neighbors=n)
#     sfs1 = SFS(knn,
#            k_features="best",
#            forward=False,
#            floating=False, 
#            verbose=0,
#            scoring='accuracy',
#            cv=10,
#            n_jobs=-1)

#     sfs1 = sfs1.fit(x, y, custom_feature_names=x.columns)
#     acc = sfs1.k_score_
#     if acc > best_acc:
#         best_acc = acc
#         print("n: ", n)
#         print("score: ", sfs1.k_score_)
#         print("beast features: ", sfs1.k_feature_names_)
#         # print("features: ", sfs1.subsets_)
#         # print("beast features: ", sfs1.k_feature_idx_)

##############################
### SFS Feature Selection ####
##############################
print("#### SFS Bwd ####")
X = x_nodoc
sfs1 = SFS(sel_alg,
           k_features="best",
           forward=False,
           floating=False, 
           verbose=0,
           scoring='accuracy',
           cv=10,
           n_jobs=-1)

sfs1 = sfs1.fit(x, y, custom_feature_names=x.columns)
# print("features: ", sfs1.subsets_)
print("SFS Accuracy Score: ", sfs1.k_score_)
# print("beast features: ", sfs1.k_feature_idx_)
print("SFS best features: ", sfs1.k_feature_names_)

print("\n\n#### SFS Fwd ####")
X = x_nodoc
sfs1 = SFS(sel_alg,
           k_features="best",
           forward=True,
           floating=False, 
           verbose=0,
           scoring='accuracy',
           cv=10,
           n_jobs=-1)

sfs1 = sfs1.fit(x, y, custom_feature_names=x.columns)
# print("features: ", sfs1.subsets_)
print("SFS Accuracy Score: ", sfs1.k_score_)
# print("beast features: ", sfs1.k_feature_idx_)
print("SFS best features: ", sfs1.k_feature_names_)


print('\a')
