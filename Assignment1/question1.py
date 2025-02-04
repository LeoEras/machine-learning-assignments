import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

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
    
    plot_train_df = pd.DataFrame()
    plot_test_df = pd.DataFrame()
    plot_error_df = pd.DataFrame()
    plot_boundary_df = pd.read_csv("datasets/2D_grid_points.csv", header=None)
    plot_train_df["x_train"] = x_train_merged[0]
    plot_train_df["y_train"] = x_train_merged[1]
    plot_test_df["x_test"] = x_test_merged[0]
    plot_test_df["y_test"] = x_test_merged[1]
    plot_train_df["class_train"] = np.where(y_train_merged == 1, "b", "g")
    plot_test_df["predicted"] = np.where(y_pred == 1, "b", "g")
    plot_test_df["real_class_test"] = np.where(y_test_merged == 1, "b", "g")
    plot_error_df = plot_test_df[plot_test_df["predicted"] != plot_test_df["real_class_test"]]
    area_predicted = knn_model.predict(plot_boundary_df)
    plot_boundary_df["area"] = area_predicted
    plot_boundary_df["area_color"] = np.where(area_predicted == 1, "b", "g")
    plt.scatter(plot_boundary_df[0], plot_boundary_df[1], marker=".", color=plot_boundary_df["area_color"]) # Background grid
    plt.scatter("x_train", "y_train", data=plot_train_df, color=plot_train_df["class_train"], label="Train set", marker="o")
    plt.scatter("x_test", "y_test", data=plot_test_df, color=plot_test_df["real_class_test"], label="Test set", marker="+")
    plt.scatter("x_test", "y_test", data=plot_error_df, color="r", label="Error prediction")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Testing KNN with k = " + str(k_value) + ", Test Error: " + str(round((1 - accuracy)* 100)) + "%")
    plt.legend()
    plt.show()

# points = pd.read_csv("datasets/2D_grid_points.sNC.csv", header=None)
# plt.scatter(points[0], points[1])
# plt.show()