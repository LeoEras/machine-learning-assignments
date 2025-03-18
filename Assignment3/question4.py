import pandas as pd
import sys
from sklearn.svm import SVC

def train_final_model(X_train, y_train):
    final_svm = SVC(kernel="rbf", C=4.2, gamma=3)
    final_svm.fit(X_train, y_train)
    return final_svm

def diagnoseDAT(Xtest, data_dir):
    """Predicts Alzheimer's diagnosis for given test data."""
    # Load datasets
    train_sDAT = pd.read_csv(f"{data_dir}/train.fdg_pet.sDAT.csv", header=None)
    train_sNC = pd.read_csv(f"{data_dir}/train.fdg_pet.sNC.csv", header=None)
    
    # Prepare training data
    y_train_sDAT = pd.Series([1] * len(train_sDAT))
    y_train_sNC = pd.Series([0] * len(train_sNC))
    X_train = pd.concat([train_sDAT, train_sNC], ignore_index=True)
    y_train = pd.concat([y_train_sDAT, y_train_sNC], ignore_index=True)

    model = train_final_model(X_train, y_train)
    ytest = model.predict(Xtest)
    return ytest

def main():
    data_directory = sys.argv[1] # Directory for the whole datasets
    Xtest_file = sys.argv[2] # For us, the second argument is the file to be used as Xtest data
    Xtest = pd.read_csv(Xtest_file)
    predictions = diagnoseDAT(Xtest, data_directory)
    print("Predictions:", predictions)

if __name__ == '__main__':
    main()
