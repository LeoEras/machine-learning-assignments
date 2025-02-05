import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import sys

def diagnoseDAT(Xtest, data_dir):
    """
    Predicts whether individuals belong to the sNC (0) or sDAT (1) group 
    based on glucose metabolism features.

    Parameters:
    - Xtest: N_test x 2 matrix of test feature vectors.
    - data_dir: Full path to the folder containing the required CSV files.

    Returns:
    - ytest: Vector of predictions (0 for sNC, 1 for sDAT).
    """

    # Load datasets
    train_sDAT = pd.read_csv(f"{data_dir}/train.sDAT.csv", header=None)
    train_sNC = pd.read_csv(f"{data_dir}/train.sNC.csv", header=None)
    
    # Prepare training data
    y_train_sDAT = pd.Series([1] * len(train_sDAT))
    y_train_sNC = pd.Series([0] * len(train_sNC))
    X_train = pd.concat([train_sDAT, train_sNC], ignore_index=True)
    y_train = pd.concat([y_train_sDAT, y_train_sNC], ignore_index=True)

    # Feature Scaling (Standardization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    Xtest_scaled = scaler.transform(Xtest)

    # Best k and distance metric based on file best_KNN_find.py
    best_k = 16
    best_metric = "infinity" # Can also use 'chebyshev' with the same k value

    # Train the best kNN classifier with weighting
    knn = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric, weights="distance")
    knn.fit(X_train_scaled, y_train)

    # Make predictions on test data
    ytest = knn.predict(Xtest_scaled)

    return ytest

def main():
    data_directory = sys.argv[1] # Directory for the whole datasets
    Xtest_file = sys.argv[2] # For us, the second argument is the file to be used as Xtest data
    Xtest = pd.read_csv(Xtest_file, header=None).values
    predictions = diagnoseDAT(Xtest, data_directory)
    print("Predictions:", predictions)


if __name__ == '__main__':
    main()