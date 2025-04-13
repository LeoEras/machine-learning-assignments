from question1 import data_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from utils import convert_img_to_arr, evaluate_model

def best_KNN(X_train, y_train):
    k_values_to_test = [i for i in range(1, 5)] # Takes too much time, the answer is 3
    param_grid = {"n_neighbors": k_values_to_test}
    knn_model = KNeighborsClassifier()

    grid_search = GridSearchCV(knn_model, param_grid, cv=5, scoring="accuracy")
    grid_search.fit(X_train, y_train)

    best_NN = grid_search.best_params_["n_neighbors"]
    print(f"Best C value found: {best_NN}")
    return best_NN

def train_KNN(X_train, y_train, best_val):
    knn_model = KNeighborsClassifier(n_neighbors=best_val)
    knn_model.fit(X_train, y_train)
    return knn_model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = data_split()
    X_train_arr = convert_img_to_arr(X_train)
    X_test_arr = convert_img_to_arr(X_test)
    best_val = best_KNN(X_train_arr, y_train)
    model = train_KNN(X_train_arr, y_train, best_val)
    evaluate_model(model, X_test_arr, y_test, "KNN", "micro")