import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from utils import img_to_arr, evaluate_model, data_split
from sklearn.model_selection import train_test_split

C_SEARCH = np.logspace(-3, 3, 3) # np.logspace(-3, 3, 9)

def train_polynomial_svm(X_train, y_train, halve=False):
    print("\nTraining Polynomial SVM using Grid Search...")

    param_grid = {
        "C": C_SEARCH,  # Fewer C values for speed
        "degree": [2, 3]  # Removed degree 4, 5 for speed # "degree": [2, 3, 4] 
    }

    if halve: # To half size, otherwise it takes 4 hours to finish (13 min per pass, 18 passes)
        X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=.5, random_state=0, stratify=y_train)
        # Takes way less time, about an hour!
        # Pretty cool, huh?

    svm = SVC(kernel="poly", verbose=True, shrinking=False) # For faster speed

    # Cross-validation grid search (Reduced cv=3 for speed)
    grid_search = GridSearchCV(svm, param_grid, cv=3, scoring="accuracy", verbose=3)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_C = best_params["C"]
    best_degree = best_params["degree"]
    print(f"Best parameters found: C = {best_C}, Degree = {best_degree}")

    return best_C, best_degree

def train_final_model(X_train, y_train, best_C, best_degree):
    print("\nTraining final Polynomial SVM model with best parameters...")
    final_svm = SVC(kernel="poly", C=best_C, degree=best_degree)
    final_svm.fit(X_train, y_train)
    print("Final model training complete!")
    return final_svm

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = data_split()
    X_train_arr = img_to_arr(X_train)
    X_test_arr = img_to_arr(X_test)
    # best_val, best_deg = train_polynomial_svm(X_train_arr, y_train, True)
    # C = 5.623413251903491, Degree = 2 After an hour training
    model = train_final_model(X_train_arr, y_train, 5.623413251903491, 2)
    evaluate_model(model, X_test_arr, y_test, "SMV-Polynomial", "micro")