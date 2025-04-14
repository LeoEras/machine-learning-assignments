from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from utils import evaluate_model, data_split
from sklearn.model_selection import train_test_split

def best_KNN(X_train, y_train, use_subset=False):
    k_values_to_test = [i for i in range(1, 30)] # Takes too much time, the answer is 3
    param_grid = {"n_neighbors": k_values_to_test}
    knn_model = KNeighborsClassifier()

    # Reduce dataset size for faster testing (Optional)
    if use_subset: # To half size, otherwise it takes 4 hours to finish (13 min per pass, 18 passes)
        X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=.5, random_state=0, stratify=y_train)
        # Takes way less time, about an hour!
        # Pretty cool, huh?

    grid_search = GridSearchCV(knn_model, param_grid, cv=5, scoring="accuracy", verbose=3)
    grid_search.fit(X_train, y_train)

    best_NN = grid_search.best_params_["n_neighbors"]
    print(f"Best n_neighbors value found: {best_NN}")
    return best_NN

def train_KNN(X_train, y_train, best_val):
    knn_model = KNeighborsClassifier(n_neighbors=best_val)
    knn_model.fit(X_train, y_train)
    return knn_model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = data_split()
    # best_val = best_KNN(X_train, y_train, True) # Uncomment to suffer for 3-5 minutes. Comment next line too!!!!
    best_val = 3
    model = train_KNN(X_train, y_train, best_val)
    evaluate_model(model, X_test, y_test, "KNN", "micro")