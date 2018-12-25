#Author: bowen
import scipy.io as sio
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def train_svm_classifier_poly_kernel(X_train, y_train):
    clf = SVC(C=2.0,degree=2, kernel='poly')
    clf.fit(X_train, y_train) 
    return clf

def train_svm_classifier_gaussian_kernel(X_train, y_train):
    clf = SVC(C=2.0,kernel='rbf')
    clf.fit(X_train, y_train) 
    return clf

def test_svm_classifier(clf,X_test):
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
        
def draw_charts(accuracy_result_random,accuracy_result_entropy,number_of_round):
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    x = np.linspace(0, number_of_round, number_of_round+1)
    print(x)
    plt.plot(x, accuracy_result_random, color='darkorange',
             lw=lw, label='Accuracy in Active learning' ) 
    plt.plot(x, accuracy_result_entropy, color='k',
             lw=lw) 
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy')
    plt.title('Dataset: MMI')
    plt.legend(['Random sampling','Uncertainty-based sampling'],loc=2,fontsize=15)
    plt.savefig('MMI_accuracy.jpg')
    plt.show()

if 1:
    
    poly_kernel_result = None
    gaussian_kernel_result = None
    accuracy_poly_kernel = []
    accuracy_gaussian_kernel = []
    train_matrix_address = 'Data for Assignment 4/Multi Label Scene Data/X_train.mat'
    train_label_address = 'Data for Assignment 4/Multi Label Scene Data/y_train.mat'
    test_matrix_address = 'Data for Assignment 4/Multi Label Scene Data/X_test.mat'
    test_label_address = 'Data for Assignment 4/Multi Label Scene Data/y_test.mat'
   
    train_matrix = sio.loadmat(train_matrix_address)['X_train']
    train_label = sio.loadmat(train_label_address)['y_train']
    test_matrix = sio.loadmat(test_matrix_address)['X_test']
    test_label = sio.loadmat(test_label_address)['y_test']
    
    train_label_one_class = train_label[:,0]
    clf_poly_kernel = train_svm_classifier_poly_kernel(train_matrix,train_label_one_class)
    clf_gau_kernel = train_svm_classifier_gaussian_kernel(train_matrix,train_label_one_class)
    poly_kernel_result = test_svm_classifier(clf_poly_kernel,test_matrix)
    gaussian_kernel_result = test_svm_classifier(clf_gau_kernel,test_matrix)
    poly_kernel_result = poly_kernel_result.reshape((-1,1))
    gaussian_kernel_result = gaussian_kernel_result.reshape((-1,1))
    for i in range(1,6):
        train_label_one_class = train_label[:,i]
        clf_poly_kernel = train_svm_classifier_poly_kernel(train_matrix,train_label_one_class)
        clf_gau_kernel = train_svm_classifier_gaussian_kernel(train_matrix,train_label_one_class)
        poly_result = test_svm_classifier(clf_poly_kernel,test_matrix).reshape((-1,1))
        gau_result = test_svm_classifier(clf_gau_kernel,test_matrix).reshape((-1,1))
        poly_kernel_result = np.concatenate((poly_kernel_result,poly_result),axis = 1)
        gaussian_kernel_result = np.concatenate((gaussian_kernel_result,gau_result),axis = 1)
    for item in range(test_label.shape[0]):
        accuracy_poly = calculate_accuracy(test_label[item,:],poly_kernel_result[item,:])
        accuracy_poly_kernel.append(accuracy_poly)
        accuracy_gaussian = calculate_accuracy(test_label[item,:],gaussian_kernel_result[item,:])
        accuracy_gaussian_kernel.append(accuracy_gaussian)
        
    avg_accuracy_poly = np.mean(accuracy_poly_kernel)
    avg_accuracy_gau = np.mean(accuracy_gaussian_kernel)
    
    print('avg_accuracy_poly:' + str(avg_accuracy_poly))
    print('avg_accuracy_gau:' + str(avg_accuracy_gau))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
