from sklearn.svm import SVC
import numpy as np
from utils import flatten
import sys

def train_final_model(X_train, y_train):
    print("\nTraining final Polynomial SVM model with best parameters...")
    # best_val, best_deg = 5.623413251903491, 2
    final_svm = SVC(kernel="poly", C=5.623413251903491, degree=2)
    final_svm.fit(X_train, y_train)
    print("Final model training complete!")
    return final_svm

def data_load(data_dir):
    train_data = np.load(data_dir + "mnist_train_data.npy")
    test_labels = np.load(data_dir + "mnist_train_labels.npy")
    train_data = np.array(flatten(train_data))
    return train_data, test_labels


def classifyHandwrittenDigits(Xtest, data_dir, model_path):
    """Returns a vector of predictions with elements "0", "1", ..., "9",
    corresponding to each of the N_test test images in Xtest
    Xtest N_test x 28 x 28 matrix of test images
    data_dir full path to the folder containing the following files:
    mnist_train_data.npy, mnist_train_labels.npy
    model_path (optional) full path to a deep neural network model in the .pth (PyTorch) format
    """
    # We got a better accuracy using polySVC, no need of model_path
    X_train, y_train = data_load(data_dir)
    model = train_final_model(X_train, y_train) # Takes about a minute and a half, could use pickle for this too!
    ytest = model.predict(Xtest)
    return ytest

def main():
    data_directory = sys.argv[1] # Directory for the whole datasets
    Xtest_file = sys.argv[2] # For us, the second argument is the file to be used as Xtest data
    Xtest_np = np.load(Xtest_file)
    Xtest = np.array(flatten(Xtest_np))
    predictions = classifyHandwrittenDigits(Xtest, data_directory, None)
    print("Predictions:", predictions)

if __name__ == '__main__':
    main()