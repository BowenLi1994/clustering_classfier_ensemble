# -*- coding: utf-8 -*-
"""
@author: Yuxiang Ren
"""
import scipy.io as sio
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
def train_svm_classifier_poly_kernel(X_train, y_train):
    clf = SVC(C=1.0,degree=2, kernel='poly')
    clf.fit(X_train, y_train) 
    return clf

def train_knn_classifier(X_train, y_train):
    clf = KNeighborsClassifier(n_neighbors=7)
    clf.fit(X_train, y_train)
    return clf

def train_neural_network_classifier(X_train, y_train):
    clf = MLPClassifier(hidden_layer_sizes=(25,), random_state=1)
    clf.fit(X_train, y_train) 
    return clf

def test_classifier(clf,X_test):
    result = clf.predict(X_test)  
    return result

def calculate_accuracy(ground_truth, predict_result):
    numerator = 0
    denominator = 0
    for i in range(6):
        if ground_truth[i] == 1 or predict_result[i] == 1:
            denominator +=1
        if ground_truth[i] == 1 and predict_result[i] == 1:
            numerator += 1
    accuracy = numerator/denominator
    return accuracy

def train_ensemble_majority_voting(clf_svm, clf_knn, clf_neural_network, X_train, y_train):
    eclf = VotingClassifier(estimators=[
         ('svm', clf_svm), ('knn', clf_knn), ('neural_network', clf_neural_network)], voting='hard')
    eclf = eclf.fit(X_train, y_train)
    return eclf    

if 1:
    
    train_matrix_address = 'Data for Assignment 4/Handwritten Digits/X_train.mat'
    train_label_address = 'Data for Assignment 4/Handwritten Digits/y_train.mat'
    test_matrix_address = 'Data for Assignment 4/Handwritten Digits/X_test.mat'
    test_label_address = 'Data for Assignment 4/Handwritten Digits/y_test.mat'
   
    train_matrix = sio.loadmat(train_matrix_address)['X_train']
    train_label = sio.loadmat(train_label_address)['y_train']
    test_matrix = sio.loadmat(test_matrix_address)['X_test']
    test_label = sio.loadmat(test_label_address)['y_test']
    
    clf_svm = train_svm_classifier_poly_kernel(train_matrix,train_label)
    svm_result = test_classifier(clf_svm,test_matrix)
    svm_accuracy = accuracy_score(test_label,svm_result)
    
    clf_knn = train_knn_classifier(train_matrix,train_label)
    knn_result = test_classifier(clf_knn,test_matrix)
    knn_accuracy = accuracy_score(test_label,knn_result)
    
    clf_neural_network = train_neural_network_classifier(train_matrix,train_label)
    neural_network_result = test_classifier(clf_neural_network,test_matrix)
    neural_network_accuracy = accuracy_score(test_label,neural_network_result)
    
    clf_ensemble = train_ensemble_majority_voting(clf_svm, clf_knn, clf_neural_network, train_matrix,train_label)
    ensemble_result = test_classifier(clf_ensemble,test_matrix)
    ensemble_accuracy = accuracy_score(test_label,ensemble_result)
    
    print('svm_accuracy:' + str(svm_accuracy))
    print('knn_accuracy:' + str(knn_accuracy))
    print('neural_network_accuracy:' + str(neural_network_accuracy))
    print('ensemble_accuracy:' + str(ensemble_accuracy))
    