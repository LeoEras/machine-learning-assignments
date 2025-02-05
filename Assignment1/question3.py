import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load datasets
train_sDAT = pd.read_csv("datasets/train.sDAT.csv", header=None)
train_sNC = pd.read_csv("datasets/train.sNC.csv", header=None)
test_sDAT = pd.read_csv("datasets/test.sDAT.csv", header=None)
test_sNC = pd.read_csv("datasets/test.sNC.csv", header=None)

# Prepare training data
y_train_sDAT = pd.Series([1] * len(train_sDAT))
y_train_sNC = pd.Series([0] * len(train_sNC))
X_train = pd.concat([train_sDAT, train_sNC], ignore_index=True)
y_train = pd.concat([y_train_sDAT, y_train_sNC], ignore_index=True)

# Prepare testing data
y_test_sDAT = pd.Series([1] * len(test_sDAT))
y_test_sNC = pd.Series([0] * len(test_sNC))
X_test = pd.concat([test_sDAT, test_sNC], ignore_index=True)
y_test = pd.concat([y_test_sDAT, y_test_sNC], ignore_index=True)

# Select the best distance metric (Euclidean or Manhattan) from Question 1 & 2
#best_metric = "euclidean"  # Change to "manhattan" if Manhattan performed better
best_metric = "euclidean"

# Define k values for Model Capacity (1/k)
k_values = np.arange(1, 101)  # Explore k from 1 to 100
model_capacity = 1 / k_values

train_errors = []
test_errors = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric=best_metric)
    knn.fit(X_train, y_train)

    # Calculate training and test errors
    train_acc = accuracy_score(y_train, knn.predict(X_train))
    test_acc = accuracy_score(y_test, knn.predict(X_test))

    train_errors.append(1 - train_acc)
    test_errors.append(1 - test_acc)

# Plot Error Rate vs Model Capacity (log-scale)
plt.figure(figsize=(8, 6))
plt.plot(model_capacity, train_errors, label="Training Error", marker="o")
plt.plot(model_capacity, test_errors, label="Test Error", marker="s")
plt.xscale("log")  # Log scale for x-axis
plt.xlabel("Model Capacity (1/k)")
plt.ylabel("Error Rate")
plt.title("Error Rate vs Model Capacity (" + best_metric + ")")
plt.legend()
plt.grid()
plt.show()
