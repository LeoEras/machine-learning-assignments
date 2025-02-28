import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import sys

def predictCompressiveStrength(Xtest, data_dir):
    '''
    Returns a vector of predictions of real number values,
    corresponding to each of the N_test features vectors in Xtest
    Xtest N_test x 8 matrix of test feature vectors
    data_dir full path to the folder containing the following files:
    train.csv, test.csv
    '''
    train_df = pd.read_csv(data_dir + "/train.csv")
    # test_df = pd.read_csv(data_dir + "/test.csv")
    x_dataset = train_df[train_df.columns.drop("ConcreteCompressiveStrength_MPa_Megapascals_")]
    y_labels = train_df[["ConcreteCompressiveStrength_MPa_Megapascals_"]]
    model = Lasso(alpha=0.01624)
    model.fit(x_dataset, y_labels)

    x_test = Xtest[Xtest.columns.drop("ConcreteCompressiveStrength_MPa_Megapascals_")]
    # y_test_labels = Xtest[["ConcreteCompressiveStrength_MPa_Megapascals_"]]
    y_pred = model.predict(x_test)
    return y_pred

def main():
    data_directory = sys.argv[1] # Directory for the whole datasets
    Xtest_file = sys.argv[2] # For us, the second argument is the file to be used as Xtest data
    Xtest = pd.read_csv(Xtest_file)
    predictions = predictCompressiveStrength(Xtest, data_directory)
    print("Predictions:", predictions)

if __name__ == '__main__':
    main()