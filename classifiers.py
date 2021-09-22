# Nearest Neighbors
# RBF SVM
# Gaussian Process
# Decision Tree
# Random Forest
# Neural Net
# AdaBoost
# Naive Bayes
# QDA

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from pyriemann.classification import MDM


classifiers = [
    # KNeighborsClassifier(),
    # SVC(kernel='linear'),
    SVC(kernel='rbf', gamma='scale', C=1),
    # GaussianProcessClassifier(),
    # DecisionTreeClassifier(),
    # RandomForestClassifier(),
    # MLPClassifier(max_iter=1000),
    # AdaBoostClassifier(),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis(),
    LogisticRegression(),
    # MDM(metric='riemann')
]

# Logistic regression
# SVM
#

print(classifiers)
