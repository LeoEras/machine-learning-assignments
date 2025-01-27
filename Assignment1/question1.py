import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_sDAT = pd.read_csv("datasets/train.sDAT.csv", header=None)
train_sNC = pd.read_csv("datasets/train.sNC.csv", header=None)
test_sDAT = pd.read_csv("datasets/test.sDAT.csv", header=None)
test_sNC = pd.read_csv("datasets/test.sNC.csv", header=None)

# Data cleaning (Training data)
y_train_sDAT = pd.Series([1 for _ in range(len(train_sDAT))])
y_train_sNC = pd.Series([0 for _ in range(len(train_sNC))])
x_train_merged = pd.concat([train_sDAT, train_sNC], ignore_index=True, sort=False)
y_train_merged = pd.concat([y_train_sDAT, y_train_sNC], ignore_index=True, sort=False)

# Data cleaning (Testing data)
y_test_sDAT = pd.Series([1 for _ in range(len(test_sDAT))])
y_test_sNC = pd.Series([0 for _ in range(len(test_sNC))])
x_test_merged = pd.concat([test_sDAT, test_sNC], ignore_index=True, sort=False)
y_test_merged = pd.concat([y_test_sDAT, y_test_sNC], ignore_index=True, sort=False)


k_values_to_test = [1, 3, 5, 10, 20, 30, 40, 45, 50, 100, 150, 200]
for k_value in k_values_to_test:
    knn_model = KNeighborsClassifier(n_neighbors=k_value)
    knn_model.fit(x_train_merged, y_train_merged)

    # Calculate the accuracy of the model.
    y_pred = knn_model.predict(x_test_merged)
    accuracy = accuracy_score(y_test_merged, y_pred)
    print("Accuracy using k = {}: {:.2f}%".format(k_value, accuracy * 100))