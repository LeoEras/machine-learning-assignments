from question1 import data_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from utils import convert_img_to_arr
import numpy as np

def grid_KNN(X_train, y_train):
    k_values_to_test = [i for i in range(1, 10)]
    param_grid = {"n_neighbors": k_values_to_test}
    knn_model = KNeighborsClassifier()

    grid_search = GridSearchCV(knn_model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_C = grid_search.best_params_["n_neighbors"]
    print(f"Best C value found: {best_C}")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = data_split()
    X_train_arr = convert_img_to_arr(X_train)
    X_test_arr = convert_img_to_arr(X_test)
    grid_KNN(X_train_arr, y_train)